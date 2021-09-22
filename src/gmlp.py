import torch
import torch.nn as nn
from g_mlp_pytorch import gMLPVision
from fastai.layers import SigmoidRange


class gMLPModel(nn.Module):

    def __init__(self, sequence_length, channels, patch_size, dim, depth, heads, ff_mult, attn_dim, prob_survival, use_stock_id, stock_embedding_dims):

        super(gMLPModel, self).__init__()

        # Stock embeddings
        self.use_stock_id = use_stock_id
        self.stock_embedding_dims = stock_embedding_dims
        self.stock_embeddings = nn.Embedding(num_embeddings=113, embedding_dim=self.stock_embedding_dims)
        self.dropout = nn.Dropout(0.25)

        # gMLP
        self.gmlp = gMLPVision(
            image_size=(sequence_length, channels),
            patch_size=patch_size,
            num_classes=1,
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
            channels=1,
            attn_dim=attn_dim,
            prob_survival=prob_survival
        )
        del self.gmlp.to_logits[2]

        self.head = nn.Sequential(
            nn.Linear(256 + self.stock_embedding_dims, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, stock_ids, sequences):

        x = sequences.view(-1, 1, sequences.shape[1], sequences.shape[2])
        x = self.gmlp(x)

        if self.use_stock_id:
            embedded_stock_ids = self.stock_embeddings(stock_ids)
            x = torch.cat([x, self.dropout(embedded_stock_ids)], dim=1)

        output = self.head(x)
        return output.view(-1)
