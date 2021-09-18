import torch
import torch.nn as nn
from fastai.layers import SigmoidRange


class MLPBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate):

        super(MLPBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim, bias=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):

    def __init__(self, num_patches, hidden_dim, token_mixer_dim, token_mixer_dropout_rate, channel_mixer_dim, channel_mixer_dropout_rate):

        super(MixerBlock, self).__init__()

        self.layer_norm_token = nn.LayerNorm(hidden_dim)
        self.token_mixer = MLPBlock(num_patches, token_mixer_dim, token_mixer_dropout_rate)
        self.layer_norm_channel = nn.LayerNorm(hidden_dim)
        self.channel_mixer = MLPBlock(hidden_dim, channel_mixer_dim, channel_mixer_dropout_rate)

    def forward(self, x):

        out = self.layer_norm_token(x).transpose(1, 2)
        x = x + self.token_mixer(out).transpose(1, 2)
        out = self.layer_norm_channel(x)
        x = x + self.channel_mixer(out)

        return x


class MLPMixer(nn.Module):

    def __init__(self, sequence_length, channels, patch_size, hidden_dim, num_blocks, token_mixer_dim, token_mixer_dropout_rate, channel_mixer_dim, channel_mixer_dropout_rate, use_stock_id, stock_embedding_dims):

        super(MLPMixer, self).__init__()

        # Stock embeddings
        self.use_stock_id = use_stock_id
        self.stock_embedding_dims = stock_embedding_dims
        self.stock_embeddings = nn.Embedding(num_embeddings=113, embedding_dim=self.stock_embedding_dims)
        self.dropout = nn.Dropout(0.25)

        # Patch embeddings
        num_patches = (sequence_length // patch_size[0]) * (channels // patch_size[1])
        self.patch_embeddings = nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        # Mixer blocks
        self.mixer = nn.Sequential(
            *[
                MixerBlock(
                    num_patches=num_patches,
                    hidden_dim=hidden_dim,
                    token_mixer_dim=token_mixer_dim,
                    token_mixer_dropout_rate=token_mixer_dropout_rate,
                    channel_mixer_dim=channel_mixer_dim,
                    channel_mixer_dropout_rate=channel_mixer_dropout_rate
                ) for _ in range(num_blocks)
            ]
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.stock_embedding_dims, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, stock_ids, sequences):

        x = sequences.view(-1, 1, sequences.shape[1], sequences.shape[2])
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mixer(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)

        if self.use_stock_id:
            embedded_stock_ids = self.stock_embeddings(stock_ids)
            x = torch.cat([x, self.dropout(embedded_stock_ids)], dim=1)

        output = self.head(x)
        return output.view(-1)
