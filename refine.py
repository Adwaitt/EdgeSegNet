import torch
from torch import nn

class Refine(nn.Module):
    def __init__(self, in_channels = 13, out_channels = 193):
        super(Refine, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1, stride = 1)
        self.conv3 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x):
        y = self.conv1(x)
        y_ = self.conv2(y)
        y_ = self.relu(y_)
        y_ = self.conv3(y_)
        y_ = self.relu(y_)
        y_ = self.conv4(y_)
        y = y + y_
        return y