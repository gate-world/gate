import torch
from matplotlib import pyplot as plt
from src.tools.myplot import heatmap
a = torch.load('w84.pth')
# plt.plot(torch.sum(a, dim=0).data.numpy())
# plt.show()
temp = torch.mean(a, dim=1)
heatmap(temp.data.numpy())
