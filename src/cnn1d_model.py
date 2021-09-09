import math
import torch
import torch.nn as nn
from fastai.layers import SigmoidRange


class Conv1dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, skip_connection=False):

        super(Conv1dBlock, self).__init__()

        self.skip_connection = skip_connection
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(kernel_size,), stride=(stride,), padding=(kernel_size // 2,), padding_mode='replicate', bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), stride=(stride,), bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):

        output = self.conv_block(x)
        if self.skip_connection:
            x = self.downsample(x)
            output += x
        output = self.relu(output)

        return output


class Conv1dLayers(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, depth_scale, width_scale, skip_connection, initial=False):

        super(Conv1dLayers, self).__init__()

        depth = int(math.ceil(2 * depth_scale))
        width = int(math.ceil(out_channels * width_scale))

        if initial:
            layers = [Conv1dBlock(
                    in_channels=in_channels,
                    out_channels=width,
                    kernel_size=kernel_size,
                    stride=2,
                    skip_connection=skip_connection
            )]
        else:
            layers = [Conv1dBlock(
                in_channels=(int(math.ceil(in_channels * width_scale))),
                out_channels=width,
                kernel_size=kernel_size,
                stride=2,
                skip_connection=skip_connection
            )]

        for _ in range(depth - 1):
            layers += [Conv1dBlock(
                in_channels=width,
                out_channels=width,
                kernel_size=kernel_size,
                stride=1,
                skip_connection=skip_connection
            )]

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layers(x)


class CNN1DModel(nn.Module):

    def __init__(self, in_channels, out_channels, use_stock_id, stock_embedding_dims, alpha, beta, phi):

        super(CNN1DModel, self).__init__()

        # Stock embeddings
        self.use_stock_id = use_stock_id
        self.stock_embedding_dims = stock_embedding_dims
        self.stock_embeddings = nn.Embedding(num_embeddings=113, embedding_dim=self.stock_embedding_dims)
        self.dropout = nn.Dropout(0.25)

        # Model scaling
        depth_scale = alpha ** phi
        width_scale = beta ** phi
        self.out_channels = int(math.ceil(out_channels * width_scale))

        # Convolutional layers
        self.conv_layers1 = Conv1dLayers(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=5,
            depth_scale=depth_scale,
            width_scale=width_scale,
            skip_connection=False,
            initial=True
        )
        self.conv_layers2 = Conv1dLayers(
            in_channels=32,
            out_channels=64,
            kernel_size=7,
            depth_scale=depth_scale,
            width_scale=width_scale,
            skip_connection=False,
            initial=False
        )
        self.conv_layers3 = Conv1dLayers(
            in_channels=64,
            out_channels=128,
            kernel_size=9,
            depth_scale=depth_scale,
            width_scale=width_scale,
            skip_connection=False,
            initial=False
        )
        self.conv_layers4 = Conv1dLayers(
            in_channels=128,
            out_channels=self.out_channels,
            kernel_size=11,
            depth_scale=depth_scale,
            width_scale=width_scale,
            skip_connection=False,
            initial=False
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(144 + self.stock_embedding_dims, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, stock_ids, sequences):

        x = torch.transpose(sequences, 1, 2)
        x = self.conv_layers1(x)
        x = self.conv_layers2(x)
        x = self.conv_layers3(x)
        x = self.conv_layers4(x)
        x = self.pooling(x)
        x = x.view(-1, x.shape[1])

        if self.use_stock_id:
            embedded_stock_ids = self.stock_embeddings(stock_ids)
            x = torch.cat([x, self.dropout(embedded_stock_ids)], dim=1)

        output = self.head(x)
        return output.view(-1)
