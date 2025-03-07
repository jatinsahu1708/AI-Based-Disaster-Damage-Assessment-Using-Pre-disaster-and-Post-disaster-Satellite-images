import torch
import torch.nn as nn

class ChannelAttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 2 * channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, pre_feat, post_feat):
        combined = torch.cat([pre_feat, post_feat], dim=1)
        attn_weights = self.attention(combined)
        pre_attn, post_attn = torch.split(attn_weights, pre_feat.size(1), dim=1)
        return pre_feat * pre_attn + post_feat * post_attn


import torch
import torch.nn as nn

class DifferenceFusion(nn.Module):
    def __init__(self):
        super(DifferenceFusion, self).__init__()
        # Optionally, you could add further processing (e.g. a convolution) if needed.
    
    def forward(self, pre_feat, post_feat):
        # Compute the absolute difference between the pre and post feature maps.
        diff = torch.abs(pre_feat - post_feat)
        return diff
