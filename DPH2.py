import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift

class FrequencyAwareBranch(nn.Module):
    """Frequency domain analysis branch with window processing"""
    
    def __init__(self, window_type='hann'):
        super().__init__()
        self.window_type = window_type
        
    def _create_window_2d(self, height, width, device='cpu'):
        """Create 2D window function"""
        if self.window_type == 'hann':
            window_h = torch.hann_window(height, periodic=False, device=device)
            window_w = torch.hann_window(width, periodic=False, device=device)
        elif self.window_type == 'hamming':
            window_h = torch.hamming_window(height, periodic=False, device=device)
            window_w = torch.hamming_window(width, periodic=False, device=device)
        elif self.window_type == 'blackman':
            window_h = torch.blackman_window(height, periodic=False, device=device)
            window_w = torch.blackman_window(width, periodic=False, device=device)
        else:  # Rectangular window
            window_h = torch.ones(height, device=device)
            window_w = torch.ones(width, device=device)
        
        window_2d = window_h.unsqueeze(1) * window_w.unsqueeze(0)
        return window_2d
    
    def apply_window(self, x):
        """Apply window function to input feature"""
        B, C, H, W = x.shape
        window_2d = self._create_window_2d(H, W, x.device)
        return x * window_2d.unsqueeze(0).unsqueeze(0)
    
    def high_frequency_energy_ratio(self, x):
        """Calculate high frequency energy ratio, keep batch dimension"""
        x_windowed = self.apply_window(x)
        fft_val = fft2(x_windowed, dim=(-2, -1))
        fft_shifted = fftshift(fft_val, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)
        
        B, C, H, W = magnitude.shape
        center_h, center_w = H // 2, W // 2
        radius = min(center_h, center_w) // 3
        
        y_coords = torch.arange(H, device=x.device).float() - center_h
        x_coords = torch.arange(W, device=x.device).float() - center_w
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distance = torch.sqrt(Y**2 + X**2)
        high_freq_mask = distance > radius
        
        total_energy = torch.sum(magnitude**2, dim=(-2, -1))  # [B, C]
        high_freq_energy = torch.sum((magnitude**2) * high_freq_mask, dim=(-2, -1))  # [B, C]
        hfer = high_freq_energy / (total_energy + 1e-8)  # [B, C]
        
        # Average over channel dimension, keep batch dimension
        return torch.mean(hfer, dim=1)  # [B]
    
    def spectral_flatness(self, x):
        """Calculate spectral flatness, keep batch dimension"""
        x_windowed = self.apply_window(x)
        fft_val = fft2(x_windowed, dim=(-2, -1))
        magnitude = torch.abs(fft_val) + 1e-8
        
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude), dim=(-2, -1)))  # [B, C]
        arithmetic_mean = torch.mean(magnitude, dim=(-2, -1))  # [B, C]
        flatness = geometric_mean / arithmetic_mean  # [B, C]
        
        return torch.mean(flatness, dim=1)  # [B]
    
    def dominant_frequency_strength(self, x):
        """Calculate dominant frequency strength, keep batch dimension"""
        x_windowed = self.apply_window(x)
        fft_val = fft2(x_windowed, dim=(-2, -1))
        fft_shifted = fftshift(fft_val, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)
        
        max_vals, _ = torch.max(magnitude.view(magnitude.shape[0], magnitude.shape[1], -1), dim=-1)  # [B, C]
        return torch.mean(max_vals, dim=1)  # [B]
    
    def forward(self, x):
        """Return frequency domain features for each sample"""
        hfer = self.high_frequency_energy_ratio(x)        # [B]
        flatness = self.spectral_flatness(x)              # [B] 
        #dom_freq = self.dominant_frequency_strength(x)    # [B]
        
        return torch.stack([hfer, flatness], dim=1)  # [B, 3]

class GradientStatisticsBranch(nn.Module):
    """Gradient statistics branch with vectorized optimization"""
    
    def __init__(self):
        super().__init__()
        # Implement batch Sobel filtering with separable convolution
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
        # Initialize Sobel kernel weights
        sobel_x_weight = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                    dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y_weight = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                    dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.sobel_x.weight.data = sobel_x_weight
        self.sobel_y.weight.data = sobel_y_weight
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
    
    def compute_gradient_features(self, x):
        """Vectorized gradient feature calculation, return independent features for each sample"""
        B, C, H, W = x.shape
        
        # Process each channel separately then take average
        if C > 1:
            # Compute gradient for each channel and average results for multi-channel image
            gradients_per_channel = []
            for c in range(C):
                channel_data = x[:, c:c+1]  # [B, 1, H, W]
                gradients_per_channel.append(self._compute_single_channel_gradients(channel_data))
            
            # Average features across all channels [B, 5]
            features = torch.stack(gradients_per_channel, dim=0).mean(dim=0)
        else:
            # Direct calculation for single-channel image
            features = self._compute_single_channel_gradients(x)
        
        return features  # [B, 5]
    
    def _compute_single_channel_gradients(self, x):
        """Calculate gradients for single-channel image"""
        B, _, H, W = x.shape
        
        # Batch Sobel gradient computation
        gx = self.sobel_x(x)  # [B, 1, H, W]
        gy = self.sobel_y(x)  # [B, 1, H, W]
        
        # Gradient magnitude and orientation
        magnitude = torch.sqrt(gx**2 + gy**2)  # [B, 1, H, W]
        orientation = torch.atan2(gy, gx)      # [B, 1, H, W]
        
        # Reshape for statistical calculation
        magnitude_flat = magnitude.view(B, -1)  # [B, H*W]
        orientation_flat = orientation.view(B, -1)  # [B, H*W]
        
        # Compute statistical features for each sample
        gradient_mean = torch.mean(magnitude_flat, dim=1)  # [B]
        gradient_std = torch.std(magnitude_flat, dim=1)    # [B]
        gradient_max, _ = torch.max(magnitude_flat, dim=1) # [B]
        
        # Ratio of high-gradient pixels, use adaptive threshold per sample
        thresholds = gradient_mean.unsqueeze(1) + gradient_std.unsqueeze(1)  # [B, 1]
        high_gradient_mask = magnitude_flat > thresholds  # [B, H*W]
        high_gradient_ratio = torch.mean(high_gradient_mask.float(), dim=1)  # [B]
        
        # Gradient orientation consistency, calculated per sample
        orientation_std = torch.std(orientation_flat, dim=1)  # [B]
        
        # Stack all features [B, 5]
        features = torch.stack([
            gradient_mean, 
            gradient_std, 
            gradient_max, 
            high_gradient_ratio, 
            orientation_std
        ], dim=1)
        
        return features
    
    def forward(self, x):
        return self.compute_gradient_features(x)


class SpectralSmoothnessAnalyzer(nn.Module):
    """Spectral smoothness analyzer (batch parallel version)"""
    
    def __init__(self):
        super().__init__()
    
    def spectral_curvature_analysis(self, spectral_curves):
        """
        Batch calculation of spectral curvature features
        Args:
            spectral_curves: (N, C) N spectral curves with length C
        Returns:
            (N, 3) including [mean_abs_curvature, std_curvature, max_abs_curvature]
        """
        # First-order difference -> (N, C-1)
        diff1 = torch.diff(spectral_curves, n=1, dim=1)
        # Second-order difference -> (N, C-2)
        diff2 = torch.diff(diff1, n=1, dim=1)
        
        # Calculate absolute curvature
        abs_diff2 = torch.abs(diff2)
        
        curvature_mean = torch.mean(abs_diff2, dim=1)          # (N,)
        curvature_std = torch.std(diff2, dim=1)                # (N,)
        curvature_max = torch.max(abs_diff2, dim=1).values     # (N,)
        
        return torch.stack([curvature_mean, curvature_std, curvature_max], dim=1)  # (N, 3)

    def spectral_derivative_features(self, x):
        """
        Extract spectral derivative features (fully parallel computation)
        Args:
            x: (B, C, H, W)
        Returns:
            (B, 3) or (3,) for global batch average
        """
        B, C, H, W = x.shape
        
        # Number of sampling points
        sample_points = min(100, H * W)
        
        # Randomly sample spatial positions (B, sample_points)
        indices = torch.randint(0, H * W, (B, sample_points), device=x.device)  # (B, sample_points)

        # Flatten spatial dimensions: (B, C, H*W)
        x_flat = x.view(B, C, H * W)
        
        # Extract sampled points via advanced indexing
        # x_sampled: (B, C, sample_points)
        x_sampled = x_flat.gather(2, indices.unsqueeze(1).expand(-1, C, -1))
        
        # Transpose to (B, sample_points, C) and reshape to (B * sample_points, C)
        spectral_curves = x_sampled.transpose(1, 2).reshape(-1, C)  # (B * sample_points, C)
        
        # Batch curvature feature calculation: (B * sample_points, 3)
        curvature_features = self.spectral_curvature_analysis(spectral_curves)
        
        # Reshape back to (B, sample_points, 3)
        curvature_features = curvature_features.reshape(B, sample_points, 3)
        
        # Average over sampled points for each batch
        batch_stats = torch.mean(curvature_features, dim=1)  # (B, 3)
        
        # Uncomment below to return global batch average
        # return torch.mean(batch_stats, dim=0)  # (3,)
        
        return batch_stats  # (B, 3)

    def forward(self, x):
        return self.spectral_derivative_features(x)


class BandCorrelationAnalyzer(nn.Module):
    """Band correlation analyzer (batch parallel version)"""

    def __init__(self):
        super().__init__()

    def compute_correlation_features(self, x):
        """
        Batch calculation of band correlation features
        Args:
            x: (B, C, H, W)
        Returns:
            (B, 5) or (5,) for global average
        """
        B, C, H, W = x.shape

        # Flatten spatial dimensions -> (B, C, H*W)
        x_flat = x.view(B, C, H * W)

        # 1. Batch correlation coefficient matrix: (B, C, C)
        # Note: torch.corrcoef does not support batch input, implemented manually
        x_centered = x_flat - x_flat.mean(dim=2, keepdim=True)  # Mean subtraction
        var = torch.sum(x_centered ** 2, dim=2, keepdim=True)   # (B, C, 1)
        var = torch.clamp(var, min=1e-8)  # Avoid division by zero
        x_normalized = x_centered / torch.sqrt(var)  # (B, C, H*W)

        # Batch matrix multiplication to get correlation matrix
        correlation_matrices = torch.bmm(x_normalized, x_normalized.transpose(1, 2))  # (B, C, C)

        # 2. Handle NaN and Inf values
        invalid = torch.isnan(correlation_matrices) | torch.isinf(correlation_matrices)
        correlation_matrices = torch.where(invalid, torch.zeros_like(correlation_matrices), correlation_matrices)
        # Force diagonal elements to 1 to fix numerical errors
        eye = torch.eye(C, device=x.device).expand(B, C, C)
        correlation_matrices = torch.where(eye == 1, torch.ones_like(correlation_matrices), correlation_matrices)

        # 3. Extract adjacent band correlation: (B, C-1)
        adjacent_indices = torch.arange(C - 1, device=x.device)
        adjacent_corrs = correlation_matrices[:, adjacent_indices, adjacent_indices + 1]  # (B, C-1)

        # Mask NaN values
        adjacent_mask = ~torch.isnan(adjacent_corrs)
        valid_adjacent_corrs = torch.where(adjacent_mask, adjacent_corrs, torch.zeros_like(adjacent_corrs))

        # Calculate mean and standard deviation of valid values
        sum_adj = torch.sum(valid_adjacent_corrs * adjacent_mask, dim=1)  # (B,)
        count_adj = torch.sum(adjacent_mask, dim=1)  # (B,)
        mean_adjacent_corr = torch.where(count_adj > 0, sum_adj / count_adj, torch.zeros_like(sum_adj))

        # Standard deviation for valid entries only
        diff_adj = valid_adjacent_corrs - mean_adjacent_corr.unsqueeze(1)
        var_adj = torch.sum((diff_adj ** 2) * adjacent_mask, dim=1) / torch.clamp(count_adj, min=1)
        std_adjacent_corr = torch.sqrt(var_adj)
        std_adjacent_corr = torch.where(count_adj > 1, std_adjacent_corr, torch.zeros_like(std_adjacent_corr))

        # 4. Overall correlation statistics (ignore NaN)
        flat_corr = correlation_matrices.view(B, -1)  # (B, C*C)
        valid_corr_mask = ~torch.isnan(flat_corr)
        valid_count = torch.sum(valid_corr_mask, dim=1)  # (B,)

        valid_sum = torch.sum(flat_corr * valid_corr_mask, dim=1)
        mean_correlation = torch.where(valid_count > 0, valid_sum / valid_count, torch.zeros_like(valid_sum))

        # Standard deviation
        diff_all = flat_corr - mean_correlation.unsqueeze(1)
        var_all = torch.sum((diff_all ** 2) * valid_corr_mask, dim=1) / torch.clamp(valid_count, min=1)
        std_correlation = torch.sqrt(var_all)
        std_correlation = torch.where(valid_count > 1, std_correlation, torch.zeros_like(var_all))

        # 5. Effective rank estimation via batch SVD
        try:
            # Batch SVD decomposition
            U, S, V = torch.svd(correlation_matrices)  # S: (B, C)
            S_positive = torch.clamp(S, min=1e-8)
            total_s = torch.sum(S_positive, dim=1, keepdim=True)  # (B, 1)
            normalized_s = S_positive / total_s  # (B, C)
            log_s = torch.log(normalized_s + 1e-8)
            entropy = -torch.sum(normalized_s * log_s, dim=1)  # (B,)
            effective_rank = torch.exp(entropy)  # (B,)
        except:
            effective_rank = torch.full((B,), float(C), device=x.device)

        # 6. Combine all features
        features = torch.stack([
            mean_adjacent_corr,
            std_adjacent_corr,
            mean_correlation,
            std_correlation,
            effective_rank
        ], dim=1)  # (B, 5)

        return features  # (B, 5)

    def forward(self, x):
        return self.compute_correlation_features(x)


class DegradationTypeClassifier(nn.Module):
    """Degradation type classifier"""
    
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class FeatureEncoder(nn.Module):
    """Encode low-dimensional degradation features into high-dimensional vectors"""
    def __init__(self, input_dim=15, hidden_dim=64, output_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)  # Normalization for better stability
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, input_dim)
        return self.encoder(x)  # (B, output_dim)


class DPHNet(nn.Module):
    """Degradation Perception Head Network (DPH-Net)"""
    
    def __init__(self,dim,class_num=3):
        super().__init__()

        self.freq_branch = FrequencyAwareBranch()
        self.grad_branch = GradientStatisticsBranch()
        self.smoothness_analyzer = SpectralSmoothnessAnalyzer()
        self.correlation_analyzer = BandCorrelationAnalyzer()

        self.feature_encoder = FeatureEncoder(input_dim=15, output_dim=dim)
        self.cls = DegradationTypeClassifier(input_dim=dim, num_classes=class_num)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Input: x - Hyperspectral image (B, C, H, W)
        Output: Comprehensive degradation perception results
        """

        s = self.smoothness_analyzer(x)
        c = self.correlation_analyzer(x)

        f = self.freq_branch(x)
        
        # Gradient features
        g = self.grad_branch(x)

        features = torch.cat([s, c, f, g], dim=1)# (B,15)

        high_dim_features = self.feature_encoder(features)  # (B, encoding_dim)

        deg_cls = self.cls(high_dim_features)  # (B, class_num)

        return high_dim_features, deg_cls  # (B, D)