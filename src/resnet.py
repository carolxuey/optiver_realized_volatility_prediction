import torch
import torch.nn as nn
from fastai.layers import SigmoidRange


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, stride, dilation, padding):

        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv_block(x) + x
        x = self.relu(x)

        return x


class ResNetModel(nn.Module):

    def __init__(self, in_channels):

        super(ResNetModel, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, stride=1, dilation=1, padding=0, bias=True),
            nn.BatchNorm1d(in_channels * 2),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(channels=in_channels * 2, kernel_size=5, stride=1, dilation=1, padding=0),
            ResidualBlock(channels=in_channels * 2, kernel_size=5, stride=1, dilation=1, padding=0),
            nn.AvgPool2d(2)
        )
        self.head = nn.Sequential(
            nn.Linear(17580, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        print(x.shape)
        x = self.res_blocks(x)
        print(x.shape)
        x = self.head(x)
        return x
