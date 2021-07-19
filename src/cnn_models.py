import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout_rate):

        super(Conv1dBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):

        x = self.conv_block(x)
        return x


class CNNModel(nn.Module):

    def __init__(self, in_channels):

        super(CNNModel, self).__init__()
        self.conv_block1 = Conv1dBlock(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=0, dilation=1, dropout_rate=0)
        self.conv_block2 = Conv1dBlock(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1, dropout_rate=0)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=1)
        self.regressor = nn.Linear(16296, 1, bias=True)

    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        x = self.pooling(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        output = self.regressor(x)

        return output.view(-1)
