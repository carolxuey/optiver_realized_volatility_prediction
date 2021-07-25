import torch
import torch.nn as nn
from fastai.layers import SigmoidRange


class Conv1dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3,), stride=(1,), dilation=(1,), padding=(1,), skip_connection=False):

        super(Conv1dBlock, self).__init__()
        self.skip_connection = skip_connection

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, padding_mode='replicate', bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, padding_mode='replicate', bias=True),
            nn.BatchNorm1d(out_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        output = self.conv_block(x)
        if self.skip_connection:
            x = self.downsample(x)
            output += x
        output = self.relu(output)

        return output


class CNNModel(nn.Module):

    def __init__(self, in_channels):

        super(CNNModel, self).__init__()

        self.conv_block1 = Conv1dBlock(in_channels=in_channels, out_channels=16, skip_connection=True)
        self.conv_block2 = Conv1dBlock(in_channels=16, out_channels=32, skip_connection=True)
        self.conv_block3 = Conv1dBlock(in_channels=32, out_channels=64, skip_connection=True)
        self.conv_block4 = Conv1dBlock(in_channels=64, out_channels=32, skip_connection=True)
        self.conv_block5 = Conv1dBlock(in_channels=32, out_channels=16, skip_connection=True)
        self.conv_block6 = Conv1dBlock(in_channels=16, out_channels=8, skip_connection=True)
        self.conv_block7 = Conv1dBlock(in_channels=8, out_channels=1, skip_connection=True)
        self.pooling = nn.AvgPool2d(kernel_size=(3,), stride=(1,), padding=(1,))
        self.head = nn.Sequential(
            nn.Linear(600, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        x = self.pooling(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        x = self.conv_block3(x)
        x = self.pooling(x)
        x = self.conv_block4(x)
        x = self.pooling(x)
        x = self.conv_block5(x)
        x = self.pooling(x)
        x = self.conv_block6(x)
        x = self.pooling(x)
        x = self.conv_block7(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        output = self.head(x)

        return output.view(-1)
