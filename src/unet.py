import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):

        super(Conv1dBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.conv_block(x)
        return x


class Conv1dUpsample(nn.Module):

    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride, dilation, padding):

        super(Conv1dUpsample, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.upsample(x)
        return x


class UNetModel(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(UNetModel, self).__init__()
        self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv_block1 = Conv1dBlock(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv_block2 = Conv1dBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv_block3 = Conv1dBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv_block4 = Conv1dBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv_block5 = Conv1dBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, dilation=1, padding=1)

        self.upsample5 = Conv1dUpsample(scale_factor=2, in_channels=1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample_conv_block5 = Conv1dBlock(in_channels=1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample4 = Conv1dUpsample(scale_factor=2, in_channels=512, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample_conv_block4 = Conv1dBlock(in_channels=512, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample3 = Conv1dUpsample(scale_factor=2, in_channels=256, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample_conv_block3 = Conv1dBlock(in_channels=256, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample2 = Conv1dUpsample(scale_factor=2, in_channels=128, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample_conv_block2 = Conv1dBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv1d = nn.Conv1d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.conv_block1(x)
        x2 = self.pooling(x1)
        x2 = self.conv_block2(x2)
        x3 = self.pooling(x2)
        x3 = self.conv_block3(x3)
        x4 = self.pooling(x3)
        x4 = self.conv_block4(x4)
        x5 = self.pooling(x4)
        x5 = self.conv_block5(x5)

        d5 = self.upsample5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.upsample_conv_block5(d5)

        d4 = self.upsample4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.upsample_conv_block4(d4)

        d3 = self.upsample3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.upsample_conv_block3(d3)

        d2 = self.upsample2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.upsample_conv_block2(d2)

        d1 = self.conv1d(d2)
        print(d1)
        print(d1.shape)

        return d1
