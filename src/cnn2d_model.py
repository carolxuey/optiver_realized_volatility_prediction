import torch
import torch.nn as nn
from fastai.layers import SigmoidRange


class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), skip_connection=False):

        super(Conv2dBlock, self).__init__()
        self.skip_connection = skip_connection
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        output = self.conv_block(x)
        if self.skip_connection:
            x = self.downsample(x)
            output += x
        output = self.relu(output)

        return output


class CNN2DRegularModel(nn.Module):

    def __init__(self, in_channels):

        super(CNN2DRegularModel, self).__init__()

        self.stock_embeddings = nn.Embedding(num_embeddings=113, embedding_dim=16)
        self.conv_block1 = Conv2dBlock(in_channels=in_channels, out_channels=2, skip_connection=True)
        self.conv_block2 = Conv2dBlock(in_channels=2, out_channels=4, skip_connection=True)
        self.conv_block3 = Conv2dBlock(in_channels=4, out_channels=2, skip_connection=True)
        self.conv_block4 = Conv2dBlock(in_channels=2, out_channels=1, skip_connection=True)
        self.pooling = nn.AvgPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.linear = nn.Linear(9616, 256, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Sequential(
            nn.Linear(256, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, stock_ids, sequences):

        x = torch.transpose(sequences, 1, 2)
        x = torch.unsqueeze(x, 1)
        x = self.conv_block1(x)
        x = self.pooling(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        x = self.conv_block3(x)
        x = self.pooling(x)
        x = self.conv_block4(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        embedded_stock_ids = self.stock_embeddings(stock_ids)
        x = torch.cat([x, self.dropout(embedded_stock_ids)], dim=1)
        x = self.relu(self.linear(x))
        output = self.head(x)

        return output.view(-1)


class CNN2DNestedModel(nn.Module):

    def __init__(self, in_channels):

        super(CNN2DNestedModel, self).__init__()

        self.conv_block1 = Conv2dBlock(in_channels=in_channels, out_channels=2, skip_connection=True)
        self.conv_block2 = Conv2dBlock(in_channels=2, out_channels=4, skip_connection=True)
        self.conv_block3 = Conv2dBlock(in_channels=4, out_channels=2, skip_connection=True)
        self.conv_block4 = Conv2dBlock(in_channels=2, out_channels=1, skip_connection=True)
        self.pooling = nn.AvgPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.head = nn.Sequential(
            nn.Linear(256, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, stock_ids, sequences):

        x = torch.transpose(sequences, 1, 2)
        x = torch.unsqueeze(x, 1)
        x = self.conv_block1(x)
        x = self.pooling(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        x = self.conv_block3(x)
        x = self.pooling(x)
        x = self.conv_block4(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        output = self.head(x)

        return output.view(-1)
