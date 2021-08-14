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

        self.batch_norm = nn.BatchNorm1d(600)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.25,
            bidirectional=False,
            batch_first=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=True),
            SigmoidRange(0, 0.1)
        )

    def forward(self, sequences):

        self.sequences = self.batch_norm(sequences)
        h_n0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(self.device)
        c_n0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(self.device)
        lstm_output, (h_n, c_n) = self.lstm(sequences, (h_n0, c_n0))
        output = self.head(h_n.view(-1, self.hidden_size))

        return output
