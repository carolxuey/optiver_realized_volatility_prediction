import torch
import torch.nn as nn
import timm


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):

        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0,
            bidirectional=False,
            batch_first=True
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):

        h_n0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_n0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        lstm_output, (h_n, c_n) = self.lstm(x, (h_n0, c_n0))
        avg_pooled_features = torch.mean(lstm_output, 1)
        output = self.regressor(avg_pooled_features)

        return output


class Conv1DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout_rate):

        super(Conv1DBlock, self).__init__()
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
        self.conv_block1 = Conv1DBlock(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=7,
            stride=1,
            padding=(0, 0),
            dilation=1,
            dropout_rate=0
        )

    def forward(self, x):

        print(x)
        print(x.shape)
        x = self.conv_block1(x)
        print(x)
        print(x.shape)


        return x


class ResNetModel(nn.Module):

    def __init__(self, model_name, pretrained=False, trainable_backbone=True):

        super(ResNetModel, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if trainable_backbone is False:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.global_pool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_to_3d = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=3, bias=False)
        self.regressor = nn.Linear(in_features, 1, bias=True)

    def forward(self, x):

        batch_size = x.size(0)
        print(x.shape)
        x = self.conv_to_3d(x)
        print(x.shape)
        exit()
        features = self.backbone(x)
        pooled_features = self.pooling(features).view(batch_size, -1)
        output = self.regressor(pooled_features)

        return output
