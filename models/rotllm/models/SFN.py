import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.se(self.avg_pool(x))
        max_out = self.se(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ConvWide(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=8, stride=8):
        super(ConvWide, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        att = self.ca(x)
        x = x * att
        return x


class ConvMultiScale(nn.Module):
    def __init__(self, channel_num):
        super(ConvMultiScale, self).__init__()
        scale_channel_num = channel_num // 3
        self.downsample = nn.AvgPool1d(4, padding=1)
        self.conv3 = nn.Conv1d(channel_num, scale_channel_num, 3, 4, padding=1)
        self.conv5 = nn.Conv1d(channel_num, scale_channel_num, 5, 4, padding=2)
        self.conv7 = nn.Conv1d(channel_num, scale_channel_num, 7, 4, padding=3)
        self.norm = nn.BatchNorm1d(channel_num)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(channel_num)

    def forward(self, x):
        x0 = self.downsample(x)

        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        x = torch.cat([x3, x5, x7], dim=1)
        x = self.norm(x)
        x = self.relu(x)
        x = self.ca(x) * x

        feature_len = x0.size(2)
        x = x[:, :, :feature_len] + x0
        return x


class VibrationEncoder(nn.Module):
    def __init__(self, in_channels, feature_channels, conv_layers):
        super(VibrationEncoder, self).__init__()
        self.conv_wide = ConvWide(in_channels, feature_channels)
        conv_layers = [ConvMultiScale(feature_channels) for _ in range(conv_layers)]
        self.conv_multi = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_wide(x)
        x = self.conv_multi(x)
        return x


class SpecFoldNet(nn.Module):
    def __init__(self, fold_num=3, feature_channels=180, conv_layers=4, input_shape=(1, 24000)):
        super(SpecFoldNet, self).__init__()
        self.fold_num = fold_num
        self.sub_spec_len = input_shape[1] // fold_num
        self.encoder = VibrationEncoder(fold_num, feature_channels, conv_layers)
        self.encode_len = self.test_encoder_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.encode_len, 128),
            nn.ReLU(),
            nn.Linear(128, 15)
        )

    def test_encoder_out(self, input_shape):
        x = torch.randn(input_shape).unsqueeze(0)
        x = x.reshape(-1, self.fold_num, self.sub_spec_len)
        x = self.encoder(x)
        x = x.size(1) * x.size(2)
        return x

    def forward(self, x):
        x = x.view(x.size(0), self.fold_num, self.sub_spec_len)
        x = self.encoder(x)
        x = x.view(x.size(0), self.encode_len)
        x = self.fc(x)
        return x
