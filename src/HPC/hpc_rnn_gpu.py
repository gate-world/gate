"""
用gate-rnn的方式来实现HPC, gpu版本
但实际上并不太能提升训练速度，，，得要将batch_size、lr设置的更大才行。
另外从GPU转移到cpu的过程非常耗费时间。暂时先放弃这个代码，先用CPU的代码
24.07.20

实验结果：非常nice！
实现了准确率100%，且GRU做不到这一点。
自发学习到了位置细胞（CA1的发放率分布）——用于开启或者关闭EC5积分，以及Spliter（根据cue决定发放率）——用于指导行为
EC3和EC5的发放率表现，在连续的Epoch中相差很大，说不定就是一种探索？
目前的训练中没有引入噪声（EC3无关细胞和cue细胞的随机发放），之后需要尝试一下看看能不能帮助训练（或是影响训练）
另外也还没加入遗忘功能，即得到或者没得到Reward，都要将EC3的信息忘掉，不能让上一个trail的信息影响下一个。
24.07.22
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class HPCrnn(nn.Module):
    def __init__(self, ts=0.1, ecnum=100, ca1num=100, ca3num=100, ca3sigma=5, tracklength=100, actnum=2):
        """
        :param ts: 时间步长，单位为秒
        :param ecnum: EC3 & EC5的细胞数量相同，一一对应
        :param ca1num:
        :param actnum:
        :param ca3sigma: 注意需要比较大（>=5），否则训练可能非常缓慢
        """
        super(HPCrnn, self).__init__()
        self.ts = ts  # 时间步长
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.actnum = actnum
        self.ca3num = ca3num
        self.ca3sigma = ca3sigma
        self.tracklength = tracklength
        self.cuelocation = 0.1 * tracklength
        self.ca1bias = nn.Parameter(torch.zeros(ca1num))
        self.ec5bias = nn.Parameter(torch.zeros(ecnum))
        self.wca3ca1 = nn.Parameter(torch.rand(ca3num, ca1num) * 0.01)
        self.wec3ca1 = nn.Parameter(torch.randn(ecnum, ca1num) * 0.01)
        self.wca1ec5 = nn.Parameter(torch.randn(ca1num, ecnum) * 0.01)
        # self.wec5ec3 = nn.Parameter(torch.rand(ecnum)*0.01)
        self.wca1act = nn.Parameter(torch.randn(ca1num, actnum) * 0.01)
        self.actbias = nn.Parameter(torch.zeros(actnum))
        self.device = torch.device('cuda:0')

        nn.init.xavier_normal_(self.wca1ec5)
        nn.init.xavier_normal_(self.wca1act)
        nn.init.xavier_normal_(self.wec3ca1)

    def getCA3output(self, x):
        """

        :param x: 当前的位置
        :return: 每个ca3细胞的发放率
        """
        ca3center = torch.linspace(0, self.tracklength, self.ca3num)
        return torch.exp(-(ca3center - x) ** 2 / self.ca3sigma ** 2 / 2).to(self.device)

    def forward(self, cue, epoch=None):
        """

        :param epoch: 如果加上这个信息，则输出ec5和ca1的中间信息用于调试
        :param cue: (bs, ecnum), 在self.cueLocation处传递给所有EC3的额外输入
        :return:
        """
        bs = cue.shape[0]
        assert cue.shape[1] == self.ecnum
        ec3 = torch.zeros(bs, self.ecnum).to(self.device)
        ec5 = torch.rand(bs, self.ecnum).to(self.device)
        ca1 = torch.zeros(bs, self.ca1num).to(self.device)
        if epoch is not None:  # ci
            ec3his = torch.zeros(self.tracklength, bs, self.ecnum)
            ec5his = torch.zeros(self.tracklength, bs, self.ecnum)
            ca1his = torch.zeros(self.tracklength, bs, self.ca1num)
        for x in range(self.tracklength):
            if x == self.cuelocation:
                sensoryInput = cue
            else:
                sensoryInput = 0

            ca3 = self.getCA3output(x)
            # ca1 tuft部分必须限制大小，否则可能导致最终结果过大
            ca1 = torch.relu(
                torch.matmul(ca3, self.wca3ca1) * (1 + torch.sigmoid(torch.matmul(ec3, self.wec3ca1))) - self.ca1bias)
            ec5 = ec5 + 10 * self.ts * torch.matmul(ca1, self.wca1ec5) + self.ec5bias
            '''
            不同的EC5发放率截断方法，导致EC5范围不同，进而导致EC3持续时间不同。注意，根据Magee的结论应该EC5均值应取exp(-ts/tau)=0.96'''
            # ec5 = torch.clip(ec5, 0, 1)
            # ec5 = torch.sigmoid(7*(ec5-0.5))
            # ec5 = 0.9*ec5 + 0.1*(self.ts * torch.sigmoid(torch.matmul(ca1, self.wca1ec5) + self.ec5bias))  # 用GRU的方式防止EC5越界，但训练不出来
            # ec5 = 0.5 + 0.5 * torch.sigmoid(7 * (ec5 - 0.5))        # 让EC5范围为（0.52, 0.98), 收敛更快。最大值不能低于0.97。问题是，这样迭代ec5会全部趋向于1.
            ec5 = 0.5 + 0.5 * torch.sigmoid(4 * (ec5 - 0.5))

            # 加入EC3的随机切变
            ec3 = ec5 * ec3 + sensoryInput
            ec3 = ec3 + 0.2 * (torch.rand(bs, self.ecnum) < 0.04 * self.ts).float().to(self.device)
            # ec3 = torch.clip(ec3, 0, 1)       # 其实不需要这个，机制上已经保证了EC3的发放率在01之间。
            if epoch is not None:
                ec3his[x, :, :] = ec3
                ec5his[x, :, :] = ec5
                ca1his[x, :, :] = ca1
        actCell = torch.matmul(ca1, self.wca1act) + self.actbias
        if epoch is not None:
            cpu = torch.device('cpu')
            return actCell, ec3his.to(cpu), ec5his.to(cpu), ca1his.to(cpu)
        else:
            return actCell


def net_train(device, bs=100):
    start_time = time.time()  # 单位；秒
    # 进行网络训练
    epochNum = 100
    tracklength = 100
    ca1num = 100
    ecnum = 100
    actnum = 10
    net = HPCrnn(ca1num=ca1num, ecnum=ecnum, tracklength=tracklength, actnum=actnum)
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    cue = torch.zeros(bs, ecnum)  # 作为模型输入的线索
    cuePattern = torch.zeros(actnum, ecnum)
    for cueType in range(actnum):
        cuePattern[cueType, :] = 0.6 * (torch.rand(1, ecnum) < 0.2)  # 注意有可能有的细胞几个线索都响应
    cuePattern = cuePattern.to(device)
    ca1his = torch.zeros(epochNum, tracklength, bs, ca1num)
    ec5his = torch.zeros(epochNum, tracklength, bs, ecnum)
    ec3his = torch.zeros(epochNum, tracklength, bs, ecnum)
    losshis = torch.zeros(epochNum)
    for epoch in range(epochNum):
        cueUsed = np.random.randint(0, actnum, bs)
        cue = cuePattern[cueUsed, :].to(device)
        optimizer.zero_grad()
        # pred = net(cue)
        # 使用ec5.data.numpy()来观察
        pred, ec3, ec5, ca1 = net(cue, epoch)
        ca1his[epoch, :, :, :] = ca1
        ec5his[epoch, :, :, :] = ec5
        ec3his[epoch, :, :, :] = ec3
        loss = criterion(pred, torch.Tensor(cueUsed).long().to(device)).to(device)
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)      # 梯度截断
        optimizer.step()

        print(loss.to(torch.device('cpu')).item())
        losshis[epoch] = loss
    duration_time = time.time() - start_time
    print('总训练时间：%.2f' % duration_time)
    torch.save(net.state_dict(), 'hpc_rnn.pth')
    np.savez('cells.npz', ca1his.data.numpy(), ec5his.data.numpy(), ec3his.data.numpy())
    plt.plot(losshis.data.numpy())
    plt.show()


if __name__ == '__main__':
    bs = 1000

    # 训练网络并保存网络结构、中间运行过程
    seed = 5
    torch.manual_seed(seed)
    import random

    random.seed(seed)
    np.random.seed(seed)
    device = (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    print(device)

    net_train(device, bs)

    # # 读取网络并测试和查看权重
    # D = np.load('cells.npz')
    # ca1his, ec5his, ec3his = D['arr_0'], D['arr_1'], D['arr_2']
    # for i in range(100):
    #     myplot.heatmap(ca1his[:, :, 0, i])

    print('end')
