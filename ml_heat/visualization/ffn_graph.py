
import torch
import torchviz
from ml_heat.models.naive_ffn import SXNet
import matplotlib.pyplot as plt

model = SXNet()

x = torch.randn(1000, 289)

torchviz.make_dot(model(x), params=dict(model.named_parameters()))

plt.show()
