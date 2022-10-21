import torch
from torch import nn

from edgesegnet import Network

model = Network()
test_tensor = torch.rand((1, 3, 256, 256))
with torch.no_grad():
    y = model(test_tensor)
print(y.shape)