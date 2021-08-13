import torch
import torch.nn as nn
from fastai.layers import SigmoidRange


class TransformerRegularModel(nn.Module):

    def __init__(self, n_features, n_layers):

        super(TransformerRegularModel, self).__init__()

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_features,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(256, 1, bias=True),
            SigmoidRange(0, 0.1)
        )