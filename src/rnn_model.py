import torch
import torch.nn as nn


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
