import torch
from torch import nn

class RBN(nn.Module):
    def __init__(self, in_channels = 13, out_channels = 193):
        super(RBN, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features = in_channels)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1, stride = 1)
        self.branch_conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1)
    def forward(self, x):
        y = self.bn1(x)
        y = self.relu(y)
        yb = self.branch_conv(y)
        ym = self.conv1(y)
        ym = self.bn2(ym)
        ym = self.relu(ym)
        ym = self.conv2(ym)
        ym = self.bn2(ym)
        ym = self.relu(ym)
        ym = self.conv3(ym)
        y = yb + ym
        return ym