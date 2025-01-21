"""
GRU的复现，以及官方GRU网络的尝试。
还有一个GRU+linear分类器的网络。
官方的C代码真的训练太快了，自己的完全没法比；能用官方的就用官方的
24.07.22
"""

import torch
import torch.nn as nn

from src.mnist.mnist_util import MNISTutil


class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(GRU, self).__init__()
        self.feature_size, self.hidden_size = feature_size, hidden_size
        self.w_z = nn.Parameter(torch.Tensor(hidden_size + feature_size, hidden_size))
        self.w_r = nn.Parameter(torch.Tensor(hidden_size + feature_size, hidden_size))
        self.w = nn.Parameter(torch.Tensor(hidden_size + feature_size, hidden_size))
        self.biasZ = nn.Parameter(torch.zeros(hidden_size))
        self.biasR = nn.Parameter(torch.zeros(hidden_size))
        self.biasHbar = nn.Parameter(torch.zeros(hidden_size))
        nn.init.xavier_normal_(self.w_z, gain=0.1)
        nn.init.xavier_normal_(self.w_r, gain=0.1)
        nn.init.xavier_normal_(self.w, gain=0.1)

    def forward(self, x, h_0=None):
        """

        :param x: bs, T, feature_size
        :param h_0: bs, hidden_size
        :return: output(bs, T, hidden_size), h_n(bs, hidden_size)
        """
        bs, T = x.shape[:2]
        if h_0 is None:
            h_0 = torch.zeros(bs, self.hidden_size)
        assert bs == h_0.shape[0]
        assert self.hidden_size == h_0.shape[1]
        assert self.feature_size == x.shape[2]
        h = h_0
        output = torch.zeros(bs, T, self.hidden_size)
        for t in range(T):
            x_t = x[:, t, :]  # (bs, feature_size)
            z_t = torch.sigmoid(torch.matmul(torch.concat([h, x_t], 1), self.w_z) + self.biasZ)
            r_t = torch.sigmoid(torch.matmul(torch.concat([h, x_t], 1), self.w_r) + self.biasR)
            hbar_t = torch.tanh(torch.matmul(torch.concat([r_t * h, x_t], 1), self.w) + self.biasHbar)
            h = (1 - z_t) * h + z_t * hbar_t
            output[:, t, :] = h

        return output, h


class GRU_classify(nn.Module):
    def __init__(self, feature_size, hidden_size, class_size):
        super(GRU_classify, self).__init__()
        # self.gru = GRU(feature_size, hidden_size)     # 用自己的GRU
        self.gru = nn.GRU(input_size=feature_size, hidden_size=hidden_size, batch_first=True)  # 用官方的GRU
        self.fc = nn.Linear(hidden_size, class_size)

    def forward(self, x, h_0=None):
        _, h = self.gru(x, h_0)
        h = h.squeeze()  # 因为官方GRU的输出是(1, bs, hidden_size)，所以需要把第一个维度给挤出去。doc：https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.dynamic.GRU.html#gru
        return self.fc(h)


if __name__ == '__main__':
    # # 搞点随机数据试试先
    # bs, T = 3, 10
    # feature_size, hidden_size = 4, 5
    # x = torch.randn(bs, T, feature_size)
    # h_0 = torch.randn(bs, hidden_size)
    # label = torch.randn(bs, T, hidden_size)

    # seq_mnist试试, https://proceedings.neurips.cc/paper_files/paper/2017/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf
    bs, T = 20, 784  # 28*28
    feature_size, hidden_size = 1, 100  # 每次先输入一个像素试试。原文中使用hidden_size=100
    class_size = 10  # 十个数字
    h_0 = torch.zeros(bs, hidden_size)
    mnist = MNISTutil(batch_size=bs)

    net = GRU_classify(feature_size, hidden_size, class_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 注意如果用多个层的参数，这里的parameter必须用list包裹起来成为一个迭代器
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(2):
        running_loss = 0
        for i, data in enumerate(mnist.train_loader, 0):
            optimizer.zero_grad()
            x, label = data
            x = mnist.img2seq(x).unsqueeze(2)
            pred = net(x, h_0)
            # _, pred = torch.max(pred, dim=1)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:
                print('[epoch %5d, batch %5d] : % .5f' % (epoch, i, loss.item() / 20))
                running_loss = 0
