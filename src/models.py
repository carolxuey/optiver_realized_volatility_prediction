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
