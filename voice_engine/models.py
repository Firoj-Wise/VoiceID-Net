# Proprietary Ultra-Low Latency, High-Fidelity VoiceID-Net Architecture

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        w = torch.mean(x, dim=2, keepdim=True)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class Res2Conv(nn.Module):
    """
    Multi-scale Res2Net convolutional block with hierarchical sub-branches.
    Captures multi-scale temporal and spectral acoustic context.
    """
    def __init__(self, channels, scale=4, kernel_size=3, dilation=1):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.width, self.width, kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilation=dilation, bias=False
            ) for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x):
        chunks = torch.chunk(x, self.scale, dim=1)
        out = [chunks[0]]
        
        y = chunks[0]
        for i in range(self.scale - 1):
            if i == 0:
                y = chunks[i + 1]
            else:
                y = y + chunks[i + 1]
            y = F.relu(self.bns[i](self.convs[i](y)), inplace=True)
            out.append(y)
            
        return torch.cat(out, dim=1)


class ContextAwareBlock(nn.Module):
    """
    Context-Aware Multi-Scale Residual Block:
    Combines point-wise projection, multi-scale Res2Conv, and Squeeze-Excitation.
    """
    def __init__(self, in_channels, out_channels, scale=4, dilation=1, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.res2 = Res2Conv(out_channels, scale=scale, kernel_size=3, dilation=dilation)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = SqueezeExcitation(out_channels, reduction=reduction)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.res2(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual, inplace=True)


class AttentiveStatisticsPooling(nn.Module):
    """
    Context-Aware Attentive Statistics Pooling (ASP):
    Calculates attention-weighted Mean and Standard Deviation over temporal frames.
    """
    def __init__(self, in_channels, bottleneck_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.linear1 = nn.Conv1d(in_channels * 3, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_channels, kernel_size=1)
        self.out_dim = in_channels * 2

    def forward(self, x):
        context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
        context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
        
        x_in = torch.cat([x, context_mean, context_std], dim=1)
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=-1)
        
        mean = torch.sum(alpha * x, dim=-1)
        var = torch.sum(alpha * (x ** 2), dim=-1) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)


class VoiceIDNet(nn.Module):
    """
    VoiceID-Net: High-Fidelity Speaker Verification Network:
    - Multi-scale hierarchical acoustic representation
    - Context-Aware Attentive Statistics Pooling (ASP)
    - 192-dim high-fidelity embedding output
    - Fast, deterministic latency without dynamic routing stalls
    """
    def __init__(self, in_features=80, embed_dim=192, base_channels=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Acoustic Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, base_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Hierarchical Multi-Scale Stages
        self.stage1 = nn.Sequential(
            ContextAwareBlock(base_channels, base_channels, dilation=1),
            ContextAwareBlock(base_channels, base_channels, dilation=2),
        )
        self.stage2 = nn.Sequential(
            ContextAwareBlock(base_channels, base_channels * 2, dilation=1),
            ContextAwareBlock(base_channels * 2, base_channels * 2, dilation=2),
        )
        self.stage3 = nn.Sequential(
            ContextAwareBlock(base_channels * 2, base_channels * 4, dilation=1),
            ContextAwareBlock(base_channels * 4, base_channels * 4, dilation=2),
            ContextAwareBlock(base_channels * 4, base_channels * 4, dilation=3),
        )
        self.stage4 = nn.Sequential(
            ContextAwareBlock(base_channels * 4, base_channels * 8, dilation=1),
            ContextAwareBlock(base_channels * 8, base_channels * 8, dilation=2),
        )
        
        total_channels = base_channels * 8
        self.pooling = AttentiveStatisticsPooling(total_channels, bottleneck_dim=128)
        self.bn_pool = nn.BatchNorm1d(self.pooling.out_dim)
        self.linear = nn.Linear(self.pooling.out_dim, embed_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embed_dim)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x4

    def forward(self, x: torch.Tensor, return_feature_map: bool = False):
        feat_map = self.extract_features(x)
        pooled = self.pooling(feat_map)
        normed_pool = self.bn_pool(pooled)
        embedding = self.bn_out(self.linear(normed_pool))
        
        if return_feature_map is True:
            return embedding, feat_map
        return embedding


def create_model(embed_dim=192, base_channels=64):
    return VoiceIDNet(in_features=80, embed_dim=embed_dim, base_channels=base_channels)
