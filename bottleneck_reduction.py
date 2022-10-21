import torch
from torch import nn

class BNR(nn.Module):
    def __init__(self, in_channels = 13, out_channels = 193):
        super(BNR, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 8)
        self.conv_2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1, stride = 1)
        self.conv_3 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x):
        y = self.conv_1(x)
        y = self.relu(y)
        y = self.conv_2(y)
        y = self.relu(y)
        y = self.conv_3(y)
        return y