"""
用来查看训练好的网络权重的程序，在最后一行debug即可
24.10.12
"""

from src.DV.dv_gating import *
from src.tools.myplot import heatmap

temp = torch.load('hpc_rnn.pth')
net = temp['net']
single_loop_1 = net.loopList[0]
heatmap(single_loop_1.wec5ec3.data.numpy())
print('end')
