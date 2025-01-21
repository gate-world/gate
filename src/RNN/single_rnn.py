import math as m

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, feature_size, hidden_size, bias=True):
        super(RNN, self).__init__()
        self.featurea_size = feature_size  # 输入特征数量
        self.hidden_size = hidden_size  # 、隐层大小
        self.h = torch.Tensor(hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size + feature_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, h0):
        bs = x.size(0)  # batch size
        T = x.size(1)  # Time length
        assert self.hidden_size == h0.size(2)
        assert self.featurea_size == x.size(2)
        h = h0
        out = torch.zeros(bs, T, hidden_size)
        for t in range(T):
            h = torch.concat([h, torch.unsqueeze(x[:, t, :], 1)], 2)
            h = F.relu(torch.matmul(h, self.weight) + self.bias)
            out[:, t, :] = h.squeeze()
        return out, h


if __name__ == '__main__':
    # 随便整点随机数据试试
    # bs, T = 2, 3  # 批大小、序列时间长度
    # feature_size, hidden_size = 5, 4
    # input = torch.randn(bs, T, feature_size)
    # label = torch.randn(bs, 1, hidden_size)

    # 用sin-wave进行训练
    bs, T = 2, 30  # 批大小、序列时间长度
    feature_size, hidden_size = 2, 4

    criterion = nn.MSELoss()
    h0 = torch.randn(bs, 1, hidden_size)  # t=0时刻隐层的情况，随机初始化一个
    net = RNN(feature_size, hidden_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    omega = m.pi * 2
    for epoch in range(300):
        # 生成sin数据
        input = torch.zeros(bs, T, feature_size)
        label = torch.zeros(bs, T)
        t = torch.linspace(0, 3, T)  # 时间节点
        for i in range(bs):
            phi = np.remainder(np.random.rand(), (2 * m.pi))
            input[i, :, 0] = torch.sin(phi + t * omega)  # 第一维度的输入
            input[i, :, 1] = torch.sin(phi + (0.12352 + t) * omega)  # 第二维度的输入，稍微加上一点相位
            label[i, :] = 1 + torch.sin(phi + (0.3155123 + t) * omega)  # 作为label，再次加一些相位，注意必须大于零（因为函数为relu）

        optimizer.zero_grad()

        output, h_n = net(input, h0)
        # loss = criterion(h_n, label)
        loss = criterion(output[:, :, 0], label)  # 只取第一个元素，与sin作比较
        loss.backward()
        optimizer.step()
        print('loss: %.5f' % loss.item())

    print(output.shape)

    plt.plot(t.data, input[0, :, 0].data, 'k')
    plt.plot(t.data, label[0, :].data, 'b')
    plt.plot(t.data, output[0, :, 0].data, 'r')
    plt.legend(('input', 'label', 'net_output'))
    plt.show()
