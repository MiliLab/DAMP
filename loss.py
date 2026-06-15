import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(torch.nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.SAD_weight = 0.1
        self.FFT_weight = 0.05
        self.TVspatial_weight = 0.001
        self.TVspectral_weight = 0.001
        self.fidelity = CharbonnierLoss()
        self.sad = reconstruction_SADloss()
        self.fft = FFTLoss()
        self.spatial = TVLossSpatial()
        self.spectral = TVLossSpectral()

    def forward(self, y, gt):
        loss = self.fidelity(y, gt)
        return loss

class TVLossSpatial(torch.nn.Module):
    def __init__(self):
        super(TVLossSpatial, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return  (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class TVLossSpectral(torch.nn.Module):
    def __init__(self):
        super(TVLossSpectral, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return (c_tv / count_c) / batch_size
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class CharbonnierLoss0(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt((diff * diff) + (self.eps * self.eps))
        return loss

class reconstruction_SADloss(torch.nn.Module):
    def __init__(self):
        super(reconstruction_SADloss, self).__init__()

    def forward(self, x, y):
        abundance_loss = torch.acos(torch.cosine_similarity(x, y, dim=1))
        abundance_loss = torch.mean(abundance_loss)
        return abundance_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class FFTLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.criterion = torch.nn.L1Loss(reduction=reduction)
        
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return (self.criterion(pred_fft, target_fft))

