import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal

from SSAttention import SpaHSA as SpatialAttention

from DPH2 import DPHNet 

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class MySequential(nn.Sequential):
    def forward(self, x1):
        # Iterate through all layers in sequential order
        for layer in self:
            # Check if the layer takes two inputs (i.e., custom layers)
            if isinstance(layer, nn.Module):
                # Pass both inputs to the layer
                x1 = layer(x1)
            else:
                # For non-module layers, pass the two inputs directly
                x1 = layer(x1)
        return x1


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x 


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined
    
    def to_spatial(self, x, x_shape):
        h, w = x_shape
        amp, phase = x.chunk(2, dim=1)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        x = real + 1j * imag
        x = torch.fft.ifft2(x, s=(h, w), norm="backward").real
        return x

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SpectralAttention(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super(SpectralAttention, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = x.mean(dim=[2, 3])  # [B, C] Global average pooling
        y = y.unsqueeze(1)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * self.sigmoid(y)


class SSAM(nn.Module):
    def __init__(self, dim, rank, window_size, num_heads, mult, init_alpha=0.5, fixed=False):
        super(SSAM, self).__init__()
        
        # Projection for spatial branch
        self.proj_spa = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        
        # Projection for spectral branch (independent of spatial branch)
        self.proj_spe = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        
        # Spatial attention module
        self.body_spa = SpatialAttention(dim=rank, window_size=window_size, num_heads=num_heads, mult=mult)
        
        # Spectral attention module
        self.body_spe = SpectralAttention(dim=dim)
        
        # Learnable fusion weight to balance spatial and spectral branches
        if fixed:
            self.register_buffer('alpha', torch.tensor(init_alpha))  # Fixed weight, no gradient update
        else:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))  # Initialized to 0.5 for equal weighting

    def process(self, x):
        shortcut = x
        
        # --- Spatial branch ---
        spa = self.proj_spa[0](x)                    # [B, rank, H, W]
        spa = self.body_spa(spa)                     # Spatial attention
        spa = self.proj_spa[2](spa)                  # [B, dim, H, W]

        # --- Spectral branch ---
        spe = self.proj_spe(x)                       # [B, dim, H, W]
        spe = self.body_spe(spe)                     # Spectral attention

        # --- Fusion of two branches ---
        # Use sigmoid to constrain weight in range [0,1], controlled by alpha
        weight = torch.sigmoid(self.alpha)
        x_fused = weight * spa + (1 - weight) * spe  # Weighted fusion

        return x_fused + shortcut

    def forward(self, x):
        b, c, h, w = x.shape
        
        if b == 0:
            return x
        else:
            x = self.process(x)
            return x



class RoutingFunction(nn.Module):
    def __init__(self, dim, emb_dim, num_experts, k, complexity, use_complexity_bias: bool = True, complexity_scale: str="max"):
        super(RoutingFunction, self).__init__()
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        ) 
        self.freq_gate = nn.Linear(emb_dim, num_experts, bias=False)
        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)
        
        self.k = k
        self.tau = 1
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias

    def forward(self, x, emb):
        logits = self.gate(x) + self.freq_gate(emb)
        if self.training:
            loss_imp = self.importance_loss(logits.softmax(dim=-1))
        
        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        # Final auxiliary loss
        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = 0
        
        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)
        return gates, top_k_indices, top_k_values, aux_loss

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)
        importance = importance * (self.complexity * self.tau) if self.use_complexity_bias else importance
        imp_mean = importance.mean()
        imp_std = importance.std()
        loss_imp = (imp_std / (imp_mean + 1e-8)) ** 2
        return loss_imp

    def load_loss(self, logits, logits_noisy, noise_std):
        # Compute the noise threshold
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]
        
        # Compute the load for each expert
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )
        
        # Calculate noise required to win
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std
        
        # Probability of being above the threshold
        normal_dist = Normal(0, 1)
        p = 1. - normal_dist.cdf(noise_required_to_win)
        
        # Compute mean probability for each expert over examples
        p_mean = p.mean(dim=0)
        
        # Compute p_mean's coefficient of variation squared
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2
        
        return loss_load



class AdapterLayerSpaSpe(nn.Module):
    def __init__(self, 
                 dim: int, rank: int, num_experts: int = 4, top_k: int=1,
                 emb_dim: int=128, with_complexity: bool=False, complexity_scale: str="min"):
        super().__init__()            
        
        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.noise_eps = 1e-2
        self.num_experts = num_experts

        window_sizes = []
        num_heads = []
        mults = []
        init_alphas = [4.0, 1.0, -1.0, -4.0]

        for i in range(num_experts):
            window_sizes.append(8)
            num_heads.append(2)
            mults.append(4)

        self.experts = nn.ModuleList([
            MySequential(*[SSAM(dim, rank, window_size, num_head, mult, init_alpha=init_alpha)])
            for idx, (window_size, num_head, mult, init_alpha) in enumerate(zip(window_sizes, num_heads, mults, init_alphas))
        ])
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, emb_dim, 
            num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity, complexity_scale=complexity_scale
        )
        
    def forward(self, x, emb):
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x, emb)
        self.loss = aux_loss
                
        # routing
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_outputs = [self.experts[exp](expert_inputs[exp]) for exp in range(len(self.experts))]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            selected_experts = [self.experts[i] for i in top_k_indices.squeeze(0)]  # Select the corresponding experts
            expert_outputs = torch.stack([expert(x) for expert in selected_experts], dim=1)
            gates = gates.gather(1, top_k_indices)  
            weighted_outputs = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs 
            out = weighted_outputs.sum(dim=1)  # Sum across the top-k dimension to get the final output
            
        out = self.proj_out(out)

        return out


## Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        
        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type)
        ])
        
        self.mixer = SpatialAttention(dim = dim, window_size = 8, num_heads = num_heads, mult = ffn_expansion_factor)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.mixer(self.norms[0](x))
        x = x + self.ffn(self.norms[1](x))
        return x
    
class ConcatFusion(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x_q, x_kv):
        # x_q, x_kv: (B, C, H, W)
        out = torch.cat([x_q, x_kv], dim=1)  # B, 2C, H, W
        return self.fuse(out)

class DAMoE(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, emb_dim:int=128, with_complexity: bool=False):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        ])
        self.adapter_spaspe = AdapterLayerSpaSpe(
            dim, rank, 
            top_k=top_k, num_experts=num_experts, emb_dim=emb_dim,
            with_complexity=with_complexity, complexity_scale=complexity_scale
        )
        #self.shared = SpatialAttention(dim = dim, window_size = 8, num_heads = num_heads, mult = ffn_expansion_factor)
        self.shared = SSAM(dim, dim, window_size = 8, num_heads = num_heads, mult = ffn_expansion_factor, init_alpha=0.9, fixed = True)       
        self.mixer = ConcatFusion(dim)
        self.loss = None

    def forward(self, x, dp):        
        x_s = self.proj[0](x)
        x_a = self.proj[1](x)
        x_s = self.shared(x_s)
        x_a = self.adapter_spaspe(x_a, dp) + x_a
        self.loss = self.adapter_spaspe.loss
        return self.mixer(x_a, x_s)

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, emb_dim:int=128, with_complexity: bool=False):
        super().__init__()

        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type),
        ])
        
        self.DAMoE = DAMoE(dim, num_heads, ffn_expansion_factor, complexity_scale=complexity_scale,
                 rank=rank, num_experts=num_experts, top_k=top_k, emb_dim=emb_dim, with_complexity=with_complexity)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        
    def forward(self, x, emb=None):    
        shortcut = x
        x = self.norms[0](x)
        x = self.DAMoE(x, emb) + shortcut
        x = x + self.ffn(self.norms[1](x))
        return x, self.DAMoE.loss



######################################################################
## Encoder Residual Group
class EncoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim, num_heads, num_blocks, ffn_expansion, LayerNorm_type, bias):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                EncoderBlock(dim = dim, num_heads = num_heads, ffn_expansion_factor = ffn_expansion, bias = bias, LayerNorm_type = LayerNorm_type)
            )

    def forward(self, x):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x)
            i += 1
        return x    
    
    
    
######################################################################
## Decoder Residual Group
class DecoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim, num_heads, num_blocks, ffn_expansion, LayerNorm_type, bias, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, emb_dim = 128, with_complexity=False):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                DecoderBlock( dim = dim, num_heads = num_heads, ffn_expansion_factor = ffn_expansion, bias = bias, 
                             LayerNorm_type = LayerNorm_type, complexity_scale=complexity_scale, rank=rank, num_experts=num_experts, 
                             top_k=top_k, emb_dim=emb_dim, with_complexity = with_complexity)
            )

    def forward(self, x, emb=None):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x , loss = self.layers[i](x, emb)
            self.loss += loss
            i += 1
        return x  


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.ConvTranspose2d(n_feat, n_feat // 2, stride=2, kernel_size=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.body(x)


class DAMP(nn.Module):
    def __init__(self,
                inp_channels=31, 
                out_channels=31, 
                dim = 64,
                levels: int = 4,
                heads = [1,2,4,8],
                num_blocks = [2,4,4,6],
                num_dec_blocks = [2, 4, 4],
                ffn_expansion_factor = 4,
                num_refinement_blocks = 1,
                LayerNorm_type = 'WithBias', ## Other option 'BiasFree'
                bias = False,
                num_experts=4,
                topk=2,
                with_complexity=False,
                complexity_scale="max",
                emb_dim=128
                ):
        super(DAMP, self).__init__()
        
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        
        dims = [dim*2**i for i in range(levels)]

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
                
        # -- Encoder --        
        self.enc = nn.ModuleList([])
        for i in range(levels-1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i], 
                    num_blocks=num_blocks[i], 
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type, bias=True,),
                Downsample(dim*2**i)
                ])
            )
        
        # -- Latent --
        self.latent = EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1], 
            num_heads=heads[-1], 
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,)
        
        self.dep = DPHNet(dim = emb_dim, class_num= 5)
        # -- Decoder --
        dims = dims[::-1]
        heads = heads[::-1]
        num_dec_blocks = num_dec_blocks[::-1]
        
        self.dec = nn.ModuleList([])
        for i in range(levels-1):
            self.dec.append(nn.ModuleList([
                Upsample(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=1, bias=bias),
                DecoderResidualGroup(
                    dim = dims[i+1], num_heads = heads[i+1], num_blocks = num_dec_blocks[i], ffn_expansion = ffn_expansion_factor, 
                    LayerNorm_type = LayerNorm_type, bias = bias, complexity_scale = complexity_scale, rank = dims[i+1]//4, 
                    num_experts = num_experts, top_k = topk, emb_dim = 128, with_complexity = with_complexity),
                ])
            )

        # -- Refinement --
        heads = heads[::-1]
        self.refinement = EncoderResidualGroup(
            dim=dim,
            num_blocks=num_refinement_blocks, 
            num_heads=heads[0], 
            ffn_expansion=ffn_expansion_factor, 
            LayerNorm_type=LayerNorm_type, bias=True,)
        
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.total_loss = None
     
    def forward(self, x):
                
        feats = self.patch_embed(x)
        
        self.total_loss = 0
        enc_feats = []
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)
            enc_feats.append(feats)
            feats = downsample(feats)
        
        feats = self.latent(feats)
        deg_emb, deg_cls = self.dep(x)
        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            feats = block(feats, deg_emb)
            self.total_loss += block.loss

        feats = self.refinement(feats)
        x = self.output(feats) + x

        self.total_loss /= sum(self.num_dec_blocks)
        return x



if __name__ == '__main__':
    model = DAMP(
                inp_channels=31, 
                out_channels=31, 
                dim = 64,
                levels=4,
                heads=[4, 4, 4, 8],
                num_blocks=[2, 4, 4, 6],
                num_dec_blocks=[2, 4, 4],
                ffn_expansion_factor=4,
                num_refinement_blocks=1,
                LayerNorm_type='WithBias', ## Other option 'BiasFree'
                bias=False,
                num_experts=4,
                topk=2,
                with_complexity=True,
                complexity_scale="max",
                emb_dim=128
                ).cuda()
    print("Model created.")
    x = torch.randn(1, 31, 448, 448).cuda()
    print("Input tensor shape:", x.shape)
    output = model(x)
    print("Output tensor shape:", output.shape)
    print("Total auxiliary loss:", model.total_loss)