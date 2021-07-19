import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout_rate):

        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        output = self.dropout(x)

        return output


class CNNModel(nn.Module):

    def __init__(self, in_channels):

        super(CNNModel, self).__init__()
        self.conv_block1 = Conv1dBlock(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            dilation=1,
            dropout_rate=0
        )
        self.conv_block2 = Conv1dBlock(
            in_channels=12,
            out_channels=32,
            kernel_size=7,
            stride=1,
            padding=0,
            dilation=1,
            dropout_rate=0
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=5, stride=1)
        self.pooling2 = nn.AvgPool2d(kernel_size=5, stride=1)
        self.regressor = nn.Linear(14475, 1, bias=True)

    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        x = self.pooling1(x)
        x = self.conv_block2(x)
        x = self.pooling2(x)
        x = x.view(x.size(0), -1)
        output = self.regressor(x)

        return output.view(-1)
