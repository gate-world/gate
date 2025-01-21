"""
尝试用hpcrnn来实现累加学习。
from : hpc_rnn_order.py
24.09.02
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
    def __init__(self, ts=0.1, ecnum=100, ca1num=100, ca3num=100, ca3sigma=5, tracklength=100, actnum=2, stimunum=2):
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
        self.stimunum = stimunum  # 一个trail中有几次刺激
        self.ca3num = ca3num
        self.ca3sigma = ca3sigma
        self.tracklength = tracklength
        self.cuelocation = 0.08 * tracklength * (torch.Tensor(range(stimunum)) + 2)  # 多个刺激位置
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

        :param cue: (bs, stimunum, ecnum), 在self.cueLocation处传递给所有EC3的额外输入
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
        assert cue.shape[2] == self.ecnum
        ec3 = ec3_last
        ec5 = ec5_last
        ca1 = ca1_last
        ec3his = torch.zeros(self.tracklength, self.ecnum)  # 注意，这里只保留bs=0的细胞，以节省存储空间。
        ec5his = torch.zeros(self.tracklength, self.ecnum)
        ca1his = torch.zeros(self.tracklength, self.ca1num)
        num_ith_cue = 0  # 目前是在等待第几次刺激
        for x in range(self.tracklength):

            ca3 = self.getCA3output(x)
            # ca1 tuft部分必须限制大小，否则可能导致最终结果过大
            ca1 = torch.relu(
                torch.matmul(ca3, self.wca3ca1) * (1 + torch.sigmoid(torch.matmul(ec3, self.wec3ca1))) - self.ca1bias)
            # 注意这里取消了EC5的Bias，如果需要就只能依靠CA1产生，对训练结果没有负面影响
            ec5 = ec5 + 10 * self.ts * torch.matmul(ca1,
                                                    self.wca1ec5)  # + self.ec5bias
            '''
            不同的EC5发放率截断（递归迭代）方法，导致EC5范围不同，进而导致EC3持续时间不同。注意，根据Magee的结论应该EC5均值应取exp(-ts/tau)=0.96
            注意！！！！！代码到这里时，EC5可能会大于1 ！！！！因此考虑sigmoid之后的最大值，不能考虑sigmoid(1)！！！
            '''
            # ec5 = torch.clip(ec5, 0, 1)
            # ec5 = 0.9*ec5 + 0.1*(self.ts * torch.sigmoid(torch.matmul(ca1, self.wca1ec5) + self.ec5bias))  # 用GRU的方式防止EC5越界，但训练不出来
            ec5 = 0.69 + 0.3 * torch.sigmoid(4 * (ec5 - 0.3))  # 与y=x的交点为0.97，最大值0.99

            ec3 = ec5 * ec3  # EC3根据对应的EC5发放率进行衰减
            if num_ith_cue < self.stimunum and x == self.cuelocation[num_ith_cue]:
                ec3[cue[:, num_ith_cue]] = (1 - 0.6) * ec3[
                    cue[:, num_ith_cue]] + 0.6 * 1  # 希望刻画的是，将state=0的cue EC3，以0.6的概率转化为state=1
                num_ith_cue += 1

            # ec3 = torch.clip(ec3, 0, 1)       # 其实不需要这个，机制上已经保证了EC3的发放率在01之间。
            ec3his[x, :] = ec3[0, :]  # 只取第一个细胞进行记录
            ec5his[x, :] = ec5[0, :]
            ca1his[x, :] = ca1[0, :]
        actCell = torch.matmul(ca1, self.wca1act) + self.actbias
        return actCell, ec3his, ec5his, ca1his, ec3, ec5, ca1


def cue_gen(actnum, bs, stimunum, ecnum):
    cuePattern = torch.zeros(actnum, ecnum)  # 每个刺激对应的ec3情况
    cuePattern[0, 0:10] = torch.ones(10)
    cuePattern[1, 10:20] = torch.ones(10)
    cue = torch.zeros(bs, stimunum, ecnum)  # 准备给予模型的线索刺激
    cueList = torch.zeros(bs, stimunum)  # 本轮中的线索历史
    for stim in range(stimunum):
        cueLeftRight = (torch.rand(bs) < 0.5) + 0
        cue[:, stim, :] = cuePattern[cueLeftRight, :]
        cueList[:, stim] = cueLeftRight
    label = torch.sum(cueList, 1) > stimunum * 0.5
    return cue, label


def net_train(bs=30, lr=0.003):
    start_time = time.time()  # 单位；秒
    # 进行网络训练
    epochNum = 500  # 提前终止训练看看
    tracklength = 100
    ca1num = 100
    ecnum = 100
    actnum = 2
    stimunum = 6  # 每个trail给予几次（顺序的）线索刺激
    net = HPCrnn(ca1num=ca1num, ecnum=ecnum, tracklength=tracklength, actnum=actnum, stimunum=stimunum)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    ca1_all_his = torch.zeros(epochNum, tracklength, ca1num)
    ec5_all_his = torch.zeros(epochNum, tracklength, ecnum)
    ec3_all_his = torch.zeros(epochNum, tracklength, ecnum)
    ec3_last = torch.zeros(bs, ecnum)
    ec5_last = torch.rand(bs, ecnum)
    ca1_last = torch.zeros(bs, ca1num)
    losshis = torch.zeros(epochNum)
    for epoch in range(epochNum):
        cue, label = cue_gen(actnum, bs, stimunum, ecnum)
        optimizer.zero_grad()
        pred, ec3_this_his, ec5_this_his, ca1_this_his, ec3_last1, ec5_last1, ca1_last1 = net(cue, ec3_last, ec5_last,
                                                                                              ca1_last)
        ca1_all_his[epoch, :, :] = ca1_this_his  # epoch, x, i
        ec5_all_his[epoch, :, :] = ec5_this_his
        ec3_all_his[epoch, :, :] = ec3_this_his
        loss = criterion(pred, torch.Tensor(label).long())
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)      # 梯度截断
        optimizer.step()

        ec3_last = ec3_last1.detach()  # 虽然不知道为什么，但是这样就没有bug，用clone()的方法会有bug
        ec5_last = ec5_last1.detach()
        ca1_last = ca1_last1.detach()

        print('%d, %.5f' % (epoch, loss.item()))
        losshis[epoch] = loss
    duration_time = time.time() - start_time
    print('总训练时间：%.2f' % duration_time)

    cue_test, label_test = cue_gen(actnum, bs, stimunum, ecnum)
    ec3_last = torch.zeros(bs, ecnum)
    ec5_last = torch.rand(bs, ecnum)
    ca1_last = torch.zeros(bs, ca1num)
    net.eval()
    pred, _, _, _, _, _, _ = net(cue_test, ec3_last, ec5_last, ca1_last)
    acc = torch.mean(((torch.argmax(pred, 1) == label_test) + 0).float())
    print('accuracy: %.5f' % acc.float())
    torch.save(net.state_dict(), 'hpc_rnn.pth')
    np.savez('cells.npz', ca1_all_his.data.numpy(), ec5_all_his.data.numpy(), ec3_all_his.data.numpy())
    plt.plot(losshis.data.numpy())
    plt.title('batch size: %d, lr: %.4f' % (bs, lr))
    plt.savefig('./fig_result/_loss result.jpg')
    plt.show()


if __name__ == '__main__':
    batch_size = 50
    lr = 0.003
    mode = 'training'  # 本次运行，是训练还是观察结果
    # mode = 'viewing'

    # ——————————训练网络并保存网络结构、中间运行过程
    if mode == 'training':
        seed = 5
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
