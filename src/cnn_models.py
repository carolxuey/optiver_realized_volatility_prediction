import torch
import torch.nn as nn
from fastai.layers import SigmoidRange


class Conv1dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3,), stride=(1,), dilation=(1,), padding=(1,), skip_connection=False):

        super(Conv1dBlock, self).__init__()
        self.skip_connection = skip_connection

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        output = self.conv_block(x)
        print('x', x.shape)
        print('out', output.shape)
        if self.skip_connection:
            x = self.downsample(x)
            print('downsample x', x.shape)
            output += x
        output = self.relu(output)

        return output


class CNNModel(nn.Module):

    def __init__(self, in_channels):

        super(CNNModel, self).__init__()
        self.conv_block1 = Conv1dBlock(in_channels=in_channels, out_channels=16, skip_connection=True)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=1)
        self.head = nn.Sequential(
            nn.Linear(8925, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        #print('convblock1', x.shape)
        x = self.pooling(x)
        #print('pooling1', x.shape)
        #x = self.conv_block3(x)
        #x = self.pooling(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        output = self.head(x)

        return output.view(-1)
