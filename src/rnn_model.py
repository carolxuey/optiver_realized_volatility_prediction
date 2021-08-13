import torch
import torch.nn as nn
from fastai.layers import SigmoidRange


class RNNRegularModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):

        super(RNNRegularModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.stock_embeddings = nn.Embedding(num_embeddings=113, embedding_dim=16)
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0,
            bidirectional=False,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 16, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, stock_ids, sequences):

        sequences = self.batch_norm(sequences)
        h_n0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(self.device)
        c_n0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(self.device)
        lstm_output, (h_n, c_n) = self.lstm(sequences, (h_n0, c_n0))
        avg_pooled_features = torch.mean(lstm_output, 1)
        embedded_stock_ids = self.stock_embeddings(stock_ids)
        x = torch.cat([avg_pooled_features, self.dropout(embedded_stock_ids)], dim=1)
        output = self.head(x)

        return output
