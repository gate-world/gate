"""
用gate-rnn的方式来实现HPC
24.07.20

实验结果：非常nice！
实现了准确率100%，且GRU做不到这一点。
自发学习到了位置细胞（CA1的发放率分布）——用于开启或者关闭EC5积分，以及Spliter（根据cue决定发放率）——用于指导行为
EC3和EC5的发放率表现，在连续的Epoch中相差很大，说不定就是一种探索？
目前的训练中没有引入噪声（EC3无关细胞和cue细胞的随机发放），之后需要尝试一下看看能不能帮助训练（或是影响训练）
另外也还没加入遗忘功能，即得到或者没得到Reward，都要将EC3的信息忘掉，不能让上一个trail的信息影响下一个。
24.07.22
"""
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.tools import myplot


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
        # self.wec5ec3 = nn.Parameter(torch.rand(ecnum)*0.01)       # 因为每个EC3分别对应EC5，因此实际上是单位阵，就不初始化了
        self.wca1act = nn.Parameter(torch.randn(ca1num, actnum) * 0.01)
        self.actbias = nn.Parameter(torch.zeros(actnum))

        nn.init.xavier_normal_(self.wca1ec5)
        nn.init.xavier_normal_(self.wca1act)
        nn.init.xavier_normal_(self.wec3ca1)

    def getCA3output(self, x):
        """

        :param x: 当前的位置
        :return: 每个ca3细胞的发放率
        """
        ca3center = torch.linspace(0, self.tracklength, self.ca3num)
        return torch.exp(-(ca3center - x) ** 2 / self.ca3sigma ** 2 / 2)

    def forward(self, cue, ec3_last, ec5_last, ca1_last):
        """

        :param cue: (bs, ecnum), 在self.cueLocation处传递给所有EC3的额外输入
        :param ec3_last: (bs, ecnum), 继承自上一轮的信息，用于这一轮的细胞的初始化
        :param ec5_last:
        :param ca1_last: (bs, ca1num)
        :return:
            actCell: (bs, actnum), linear输出的结果，结合crossEntropy训练
            ec3his: (trachlength, ecnum)，本轮的EC3发放率历史，只取第一个细胞进行记录
            ec3:    (bs, ecnum), 本轮最后ec3的发放率，可以作为下一轮epoch EC3的初始值
        """
        cue = cue.bool()
        bs = cue.shape[0]
        assert cue.shape[1] == self.ecnum
        ec3 = ec3_last
        ec5 = ec5_last
        ca1 = ca1_last
        ec3his = torch.zeros(self.tracklength, self.ecnum)  # 注意，这里只保留bs=0的细胞，以节省存储空间。
        ec5his = torch.zeros(self.tracklength, self.ecnum)
        ca1his = torch.zeros(self.tracklength, self.ca1num)
        for x in range(self.tracklength):

            ca3 = self.getCA3output(x)
            # ca1 tuft部分必须限制大小，否则可能导致最终结果过大; Basal部分的输入是0~1之间，Tuft部分的增益是1~2之间
            ca1 = torch.relu(
                torch.matmul(ca3, self.wca3ca1)
                * (1 + torch.sigmoid(torch.matmul(ec3, self.wec3ca1)))
                - self.ca1bias)
            # 注意这里取消了EC5的Bias，如果需要就只能依靠CA1产生，对训练结果没有负面影响
            ec5 = ec5 + 10 * self.ts * torch.matmul(ca1, self.wca1ec5)
            '''
            不同的EC5发放率截断（递归迭代）方法，导致EC5范围不同，进而导致EC3持续时间不同。注意，根据Magee的结论应该EC5均值应取exp(-ts/tau)=0.96
            注意！！！！！代码到这里时，EC5可能会大于1 ！！！！因此考虑sigmoid之后的最大值，不能考虑sigmoid(1)！！！
            '''
            # ec5 = torch.clip(ec5, 0, 1)
            # ec5 = 0.9*ec5 + 0.1*(self.ts * torch.sigmoid(torch.matmul(ca1, self.wca1ec5) + self.ec5bias))  # 用GRU的方式防止EC5越界，但训练不出来
            ec5 = 0.69 + 0.3 * torch.sigmoid(4 * (ec5 - 0.3))  # 与y=x的交点为0.97，最大值0.99

            ec3 = ec5 * ec3  # EC3根据对应的EC5发放率进行衰减
            if x == self.cuelocation:
                ec3[cue] = (1 - 0.6) * ec3[cue] + 0.6 * 1  # 希望刻画的是，将state=0的cue EC3，以0.6的概率转化为state=1

            # 加入EC3的随机切变
            is_ec3_noised = (torch.rand(bs, self.ecnum) < 0.04 * self.ts)  # 本轮中每个EC3细胞是否添加noise
            ec3[is_ec3_noised] = 0.5 * ec3[is_ec3_noised] + 0.5 * 0.6  # 用以保证noised之后的EC3发放率在0.6以下
            # ec3 = torch.clip(ec3, 0, 1)       # 其实不需要这个，机制上已经保证了EC3的发放率在01之间。
            ec3his[x, :] = ec3[0, :]  # 只取第一个细胞进行记录
            ec5his[x, :] = ec5[0, :]
            ca1his[x, :] = ca1[0, :]
        actCell = torch.matmul(ca1, self.wca1act) + self.actbias
        return actCell, ec3his, ec5his, ca1his, ec3, ec5, ca1


def net_train(bs=30, lr=0.003):
    start_time = time.time()  # 单位；秒
    # 进行网络训练
    epochNum = 1500  # 提前终止训练看看
    tracklength = 100
    ca1num = 100
    ecnum = 100
    actnum = 10
    cue_ratio = 0.1  # 每个线索激活多少比例的EC3细胞
    net = HPCrnn(ca1num=ca1num, ecnum=ecnum, tracklength=tracklength, actnum=actnum)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    cue = torch.zeros(bs, ecnum)  # 作为模型输入的线索
    cuePattern = torch.zeros(actnum, ecnum)
    for cueType in range(actnum):
        cuePattern[cueType, :] = (torch.rand(1, ecnum) < cue_ratio)  # 每个细胞是否作为本轮被激活的cue细胞。注意有可能有的细胞几个线索都响应
    ca1_all_his = torch.zeros(epochNum, tracklength, ca1num)
    ec5_all_his = torch.zeros(epochNum, tracklength, ecnum)
    ec3_all_his = torch.zeros(epochNum, tracklength, ecnum)
    ec3_last = torch.zeros(bs, ecnum)
    ec5_last = torch.rand(bs, ecnum)
    ca1_last = torch.zeros(bs, ca1num)
    losshis = torch.zeros(epochNum)
    for epoch in range(epochNum):
        cueUsed = np.random.randint(0, actnum, bs)
        cue = cuePattern[cueUsed, :]  # 注意，这个操作之后似乎cue变量就变成了0和1的矩阵，需要用*.bool()还原回布尔变量
        optimizer.zero_grad()
        pred, ec3_this_his, ec5_this_his, ca1_this_his, ec3_last1, ec5_last1, ca1_last1 = net(cue, ec3_last, ec5_last,
                                                                                              ca1_last)
        ca1_all_his[epoch, :, :] = ca1_this_his  # epoch, x, i
        ec5_all_his[epoch, :, :] = ec5_this_his
        ec3_all_his[epoch, :, :] = ec3_this_his
        loss = criterion(pred, torch.Tensor(cueUsed).long())
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)      # 梯度截断
        optimizer.step()

        ec3_last = ec3_last1.detach()  # 虽然不知道为什么，但是这样就没有bug，用clone()的方法会有bug
        ec5_last = ec5_last1.detach()
        ca1_last = ca1_last1.detach()

        print('%d, %.5f' % (epoch, loss.item()))
    duration_time = time.time() - start_time
    print('总训练时间：%.2f' % duration_time)
    torch.save(net.state_dict(), 'hpc_rnn.pth')
    np.savez('cells.npz', ca1_all_his.data.numpy(), ec5_all_his.data.numpy(), ec3_all_his.data.numpy())
    plt.plot(losshis.data.numpy())
    plt.title('batch size: %d, lr: %.4f' % (bs, lr))
    plt.savefig('./fig_result/_loss result.jpg')
    plt.show()


if __name__ == '__main__':
    batch_size = 30
    lr = 0.003
    mode = 'training'  # 本次运行，是训练还是观察结果
    # mode = 'viewing'

    # ——————————训练网络并保存网络结构、中间运行过程
    if mode == 'training':
        seed = 4
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.makedirs('./fig_result', exist_ok=True)  # 保证目录的存在
        net_train(batch_size, lr)

    # ——————————读取网络并测试和查看权重
    if mode == 'viewing':
        D = np.load('cells.npz')
        os.makedirs('./fig_result', exist_ok=True)  # 保证目录的存在
        ca1his, ec5his, ec3his = D['arr_0'], D['arr_1'], D['arr_2']

        history = ca1his  # 只选择一类细胞进行绘制

        # 绘制最后一个epoch的最大发放率分布，eg：用来检查CA1是否过分稀疏，以绘制Histogram的方式
        last_ca1_rate = history[-1, :, :]  # x, index
        last_ca1_max_rate = np.amax(last_ca1_rate, axis=0)
        silence_cell_threshold = 0.1  # 静息细胞的判定条件
        print('silence cell num: % d' % np.sum(last_ca1_max_rate < silence_cell_threshold))  # 静息细胞的数量，
        last_ca1_rate = last_ca1_rate[:, last_ca1_max_rate > silence_cell_threshold]  # 删掉最大发放率太低的那些静息细胞
        last_ca1_max_rate_location = np.argmax(last_ca1_rate, axis=0)
        plt.hist(last_ca1_max_rate_location, bins=30)
        plt.title('last Epoch, distribution of Cell peak location')
        plt.savefig('./fig_result/_distrib of peak.jpg', )
        plt.show()

        # 绘制每个细胞的发放率情况, 并保存为图片
        plt.close()
        for i in range(100):
            mat = history[:, :, i]  # epoch, x
            plt.title('cell %d, max=%.5f' % (i, np.amax(mat)))
            myplot.heatmap(mat, isShow=False)
            plt.savefig('./fig_result/%d.jpg' % i)
            plt.close()
            print('cell fig %d saved. ' % i)

    print('end')
