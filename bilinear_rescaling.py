import torch
from torch import nn

class BR(nn.Module):
    def __init__(self, scale_factor):
        super(BR, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    def forward(self, x):
        y = self.upsample(x)
        return y