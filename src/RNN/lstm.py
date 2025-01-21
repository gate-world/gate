import torch
import torch.nn as nn

from src.mnist.mnist_util import MNISTutil


class LSTM(nn.Module):
    def __init__(self, feature_size, hidden_size):
        """

        :param feature_size: 输入的x变量维度
        :param hidden_size: 隐变量的维度，注意cell的size与之相等
        """
        super(LSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.WeightX = nn.Parameter(torch.Tensor(feature_size, 4 * hidden_size))  # i,f,g,o整合起来对于x的权重
        self.WeightH = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))  # i,f,g,o整合起来对于h的权重
        self.BiasX = nn.Parameter(torch.zeros(4 * hidden_size))
        self.BiasH = nn.Parameter(torch.zeros(4 * hidden_size))

        nn.init.xavier_normal_(self.WeightX, gain=0.1)
        nn.init.xavier_normal_(self.WeightH, gain=0.1)

    def forward(self, x, h_0=None):
        """

        :param x: (batch_size, time_length, feature_size)
        :param h_0: (batch_size, hidden_size)。没有人为指定的时候就直接用全0张量。
        :return: output (batch_size, time_length, hidden_size) , h_n   # output[:,-1,:] = h_n
        """
        bs, T = x.shape[:2]
        assert x.shape[2] == self.feature_size
        if h_0 is None:
            h_0 = torch.zeros(bs, self.hidden_size)
        assert x.shape[0] == h_0.shape[0]
        h = h_0
        c_t = torch.zeros(bs, self.hidden_size)
        output = torch.zeros(bs, T, self.hidden_size)
        for t in range(T):
            ifgo_linear = torch.matmul(x[:, t, :], self.WeightX) + self.BiasX \
                          + torch.matmul(h, self.WeightH) + self.BiasH  # bs, 4*cell_size
            i_t = torch.sigmoid(ifgo_linear[:, :self.hidden_size])  # 这几个变量都是 （bs, self.hidden_size)
            f_t = torch.sigmoid(ifgo_linear[:, self.hidden_size:2 * self.hidden_size])
            g_t = torch.tanh(ifgo_linear[:, self.hidden_size * 2:self.hidden_size * 3])
            o_t = torch.sigmoid(ifgo_linear[:, self.hidden_size * 3:])
            c_t = f_t * c_t + i_t * g_t
            h = o_t * torch.tanh(c_t)
            output[:, t, :] = h

        return output, h


class LSTM_classify(nn.Module):
    def __init__(self, feature_size, hidden_size, class_size):
        super(LSTM_classify, self).__init__()
        # self.lstm = LSTM(feature_size, hidden_size)
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, class_size)

    def forward(self, x, h_0=None):
        # 用我自己的lstm
        # _, h = self.lstm(x, h_0)

        # 用官方的LSTM
        _, h = self.lstm(x)
        h, _ = h
        return self.fc(h.squeeze())


if __name__ == '__main__':

    # 随便整点随机数据试试
    # bs, T = 6, 10
    # feature_size, hidden_size = 5, 3
    # input = torch.randn(bs, T, feature_size)
    # label = torch.randn(bs, hidden_size)

    # # 用sin-wave进行训练
    # bs, T = 2, 30  # 批大小、序列时间长度
    # feature_size, hidden_size = 2, 4
    # omega = m.pi * 2
    # net = LSTM(feature_size, hidden_size)
    # criterion = torch.nn.MSELoss()

    # seq_mnist试试, https://proceedings.neurips.cc/paper_files/paper/2017/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf
    bs, T = 20, 784  # 28*28
    feature_size, hidden_size = 1, 100  # 每次先输入一个像素试试。原文中使用hidden_size=100
    class_size = 10  # 十个数字
    mnist = MNISTutil(batch_size=bs)

    net = LSTM_classify(feature_size, hidden_size, class_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):

        # # 生成sin数据
        # input = torch.zeros(bs, T, feature_size)
        # label = torch.zeros(bs, T)
        # t = torch.linspace(0, 3, T)  # 时间节点
        # for i in range(bs):
        #     phi = np.remainder(np.random.rand(), (2 * m.pi))
        #     input[i, :, 0] = torch.sin(phi + t * omega)  # 第一维度的输入
        #     input[i, :, 1] = torch.sin(phi + (0.12352 + t) * omega)  # 第二维度的输入，稍微加上一点相位
        #     label[i, :] = 0.1+0.4*(1 + torch.sin(phi + (0.3155123 + t) * omega))  # 作为label，再次加一些相位，注意必须大于零小于1（因为函数为sigmoid）
        running_loss = 0
        for i, data in enumerate(mnist.train_loader, 0):
            # 进行网络训练
            optimizer.zero_grad()
            # output, h_n = net(input, h_0)
            # loss = criterion(label, h_n)
            # loss = criterion(output[:, :, 0], label)  # 只取第一个元素，与sin作比较

            x, label = data
            x = mnist.img2seq(x).unsqueeze(2)
            pred = net(x)
            # _, pred = torch.max(pred, dim=1)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:
                print('[epoch %5d, batch %5d] : % .5f' % (epoch, i, loss.item() / 20))
                running_loss = 0
            # print('loss: %.5f' % loss.item())

    # # 绘制sin-wave训练结果
    # plt.plot(t.data, input[0, :, 0].data, 'k')
    # plt.plot(t.data, label[0, :].data, 'b')
    # plt.plot(t.data, output[0, :, 0].data, 'r')
    # plt.legend(('input', 'label', 'net_output'))
    # plt.show()
