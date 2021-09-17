import torch
import torch.nn as nn
from fastai.layers import SigmoidRange

from attention import SelfAttention


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, use_stock_id, stock_embedding_dims):

        super(RNNModel, self).__init__()

        # Stock embeddings
        self.use_stock_id = use_stock_id
        self.stock_embedding_dims = stock_embedding_dims
        self.stock_embeddings = nn.Embedding(num_embeddings=113, embedding_dim=self.stock_embedding_dims)
        self.dropout = nn.Dropout(0.25)

        # Recurrent neural network
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0,
            bidirectional=False,
            batch_first=True
        )

        # Sequence self attention
        self.attention = SelfAttention(attention_size=self.hidden_size)

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size + self.stock_embedding_dims, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, stock_ids, sequences):

        h_n0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(self.device)
        gru_output, h_n = self.gru(sequences, h_n0)
        attn_output, scores = self.attention(gru_output)

        if self.use_stock_id:
            embedded_stock_ids = self.stock_embeddings(stock_ids)
            x = torch.cat([attn_output, self.dropout(embedded_stock_ids)], dim=1)

        output = self.head(x)
        return output.view(-1)
