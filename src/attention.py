import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, attention_size, batch_first=False):

        super(SelfAttention, self).__init__()

        self.attention_weights = nn.Parameter(torch.FloatTensor(attention_size))
        nn.init.uniform_(self.attention_weights.data, -0.005, 0.005)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.batch_first = batch_first

    def forward(self, x):

        attentions = self.relu(x.matmul(self.attention_weights))
        attentions = self.softmax(attentions)
        weighted = torch.mul(x, attentions.unsqueeze(-1).expand_as(x))
        representations = weighted.sum(1).squeeze()

        return representations, attentions
