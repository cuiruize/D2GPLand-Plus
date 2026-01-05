import torch
from torch import nn
from mamba_ssm.modules.mamba_simple import Mamba


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CrossMamba(nn.Module):
    def __init__(self, c, dim_s=None):
        super(CrossMamba, self).__init__()
        self.cross_mamba_spatial = Mamba(c, bimamba_type="v3")

        # ablation
        # dim_s = None

        if dim_s:
            self.cross_mamba_channel = Mamba(dim_s, bimamba_type="v3")

            self.ffn = nn.Sequential(
                nn.Conv2d(c * 2, c * 4, kernel_size=1),
                nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1, groups=c),
                nn.Conv2d(c * 4, c, kernel_size=1)
            )
            self.conv_norm = LayerNorm2d(c * 2)
        else:
            self.ffn = nn.Sequential(
                nn.Conv2d(c, c * 2, kernel_size=1),
                nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1, groups=c),
                nn.Conv2d(c * 2, c, kernel_size=1)
            )
            self.conv_norm = LayerNorm2d(c)
        self.spatial_norm1 = nn.LayerNorm(c)
        self.channel_norm1 = nn.LayerNorm(c * 2)
        self.spatial_norm2 = nn.LayerNorm(c)
        self.channel_norm2 = nn.LayerNorm(c * 2)

    def forward(self, rgb, residual, depth, feat_dim, neck=False):
        residual = rgb + residual
        global_f_spatial = self.cross_mamba_spatial(self.spatial_norm1(residual), extra_emb=self.spatial_norm2(depth))

        B, HW, C = global_f_spatial.shape
        spatial_fuse = (global_f_spatial + residual).transpose(1, 2).view(B, C, feat_dim, feat_dim)
        if neck:
            global_f_channel = self.cross_mamba_channel(self.channel_norm1(residual.permute(0, 2, 1)),
                                                        extra_emb=self.channel_norm2(depth.permute(0, 2, 1)))
            channel_fuse = (global_f_channel + residual.transpose(1, 2)).view(B, C, feat_dim, feat_dim)

            out = torch.cat((spatial_fuse, channel_fuse), dim=1)
        else:
            out = spatial_fuse

        out = self.ffn(self.conv_norm(out))
        return out, residual
