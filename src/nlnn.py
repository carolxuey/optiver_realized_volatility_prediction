import torch
import torch.nn as nn
from torch.nn import functional as F


class NonLocalBlock1d(nn.Module):

    def __init__(self, in_channels, inter_channels=None, mode='embedded'):

        super(NonLocalBlock1d, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.mode = mode

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=(1,),
            stride=(1,),
            padding=0
        )
        self.W_z = nn.Sequential(
            nn.Conv1d(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=(1,),
                stride=(1,),
                padding=0
            ),
            nn.BatchNorm1d(self.in_channels)
        )
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

        if self.mode == 'embedded' or self.mode == 'dot' or self.mode == 'concatenate':

            self.theta = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=(1,),
                stride=(1,),
                padding=0
            )
            self.phi = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=(1,),
                stride=(1,),
                padding=0
            )

        if self.mode == 'concatenate':

            self.W_f = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inter_channels * 2,
                    out_channels=1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                    bias=False
                ),
                nn.ReLU()
            )

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == 'gaussian':

            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == 'embedded' or self.mode == 'dot':

            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == 'concatenate':

            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == 'gaussian' or self.mode == 'embedded':
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == 'dot' or self.mode == 'concatenate':
            N = f.size(-1)
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        z = W_y + x

        return z
