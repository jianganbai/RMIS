import torch
import torch.nn as nn

from typing import Optional


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.se(self.avg_pool(x))
        max_out = self.se(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class ConvWide(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")

        b = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, b, 1, 4, padding=0)
        self.conv3 = nn.Conv1d(in_channels, b, 3, 4, padding=1)
        self.conv5 = nn.Conv1d(in_channels, b, 5, 4, padding=2)
        self.conv7 = nn.Conv1d(in_channels, b, 7, 4, padding=3)

        self.norm = nn.BatchNorm1d(b * 3)
        self.act = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(b * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        merged = torch.cat([x3, x5, x7], dim=1)
        merged = self.norm(merged)
        merged = self.act(merged)
        merged = self.ca(merged) * merged

        return torch.cat([x1, merged], dim=1)


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_query = ConvWide(1, 60, 8, 8)
        self.conv_ref = ConvWide(1, 8, 8, 8)
        self.conv_res = ConvWide(1, 60, 8, 8)

        self.conv = nn.Sequential(
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128),
        )

    def forward(self, x):
        query = x[:, :1, :]
        ref = x[:, 1:2, :]
        res = query - ref

        q = self.conv_query(query)
        r = self.conv_ref(ref)
        s = self.conv_res(res)

        x = torch.cat([q, r, s], dim=1)
        return self.conv(x)


class SingleInputEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = FeatureEncoder()

    def forward(self, query: torch.Tensor, ref: Optional[torch.Tensor] = None):
        if query.ndim == 2:
            query = query.unsqueeze(1)

        if ref is None:
            ref = torch.zeros_like(query)
        else:
            if ref.ndim == 2:
                ref = ref.unsqueeze(1)
            if ref.size(-1) != query.size(-1):
                min_len = min(query.size(-1), ref.size(-1))
                query = query[..., :min_len]
                ref = ref[..., :min_len]

        x = torch.cat([query, ref], dim=1)
        return self.feature_encoder(x)
