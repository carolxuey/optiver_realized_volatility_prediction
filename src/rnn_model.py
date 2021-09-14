import torch
import torch.nn as nn
from fastai.layers import SigmoidRange


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
            dropout=0.1,
            bidirectional=False,
            batch_first=True
        )
        for layer_parameters in self.gru._all_weights:
            for parameter in layer_parameters:
                if 'weight' in parameter:
                    nn.init.kaiming_normal_(self.gru.__getattr__(parameter))

        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size * 2)

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + self.stock_embedding_dims, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, stock_ids, sequences):

        h_n0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(self.device)
        gru_output, h_n = self.gru(sequences, h_n0)
        avg_pooled_output = torch.mean(gru_output, 1)
        max_pooled_output, _ = torch.max(gru_output, 1)
        x = torch.cat([avg_pooled_output, max_pooled_output], dim=1)
        x = self.batch_norm(self.relu(self.fc(x)))

        if self.use_stock_id:
            embedded_stock_ids = self.stock_embeddings(stock_ids)
            x = torch.cat([x, self.dropout(embedded_stock_ids)], dim=1)

        output = self.head(x)
        return output.view(-1)
