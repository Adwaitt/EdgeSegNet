import torch
from torch import nn

from bilinear_rescaling import BR
from bottleneck_reduction import BNR
from refine import Refine
from residual_bottleneck import RBN

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels = 3, out_channels = 13, kernel_size = 7, stride = 2, padding = 3)
        self.conv_2 = nn.Conv2d(in_channels = 217, out_channels = 1, kernel_size = 1, stride = 1)
        self.mpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.bnm = BNR()
        self.br_1 = BR(2)
        self.br_2 = BR(4)
        self.rem = RBN()
        self.rm_1 = Refine(in_channels = 193, out_channels = 193)
        self.rm_2 = Refine(in_channels = 193, out_channels = 217)
        self.rm_3 = Refine(in_channels = 13, out_channels = 217)
        self.rm_4 = Refine(in_channels = 217, out_channels = 217)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.conv_1(x)
        y_branch = self.mpool(y)
        y_sub = self.rem(y_branch)
        y_sub = self.rm_1(y_sub)
        y_branch = self.rm_3(y_branch)
        y = self.bnm(y)
        y = self.br_1(y)
        y = y + y_sub
        y = self.rm_2(y)
        y = self.br_1(y)
        y = y + y_branch
        y = self.rm_4(y)
        y = self.br_2(y)
        y = self.conv_2(y)
        y = self.sigmoid(y)
        return y



