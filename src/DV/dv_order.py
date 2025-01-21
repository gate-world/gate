"""
引入多层DV轴结构，用于进行信息的选择性存储（routing）

从结果上来看，确实是Dorsal部分展现出较小的位置场，且全都是在线索处的Splitter；ventral部分展现出更加多种多样的位置场
这点和生物上一模一样
24.09.04
from：Evidence2.py
"""
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.tools import myplot


class HPCSingleLoop(nn.Module):
    def __init__(self, ts=0.1, ecnum=100, ca1num=100, ca3num=100):
        """
        :param ts: 时间步长，单位为秒
        :param ecnum: EC3 & EC5的细胞数量相同，一一对应
        :param ca1num:
        :param actnum: 这里用于比较多少个位置（的气味对应的顺序）
        :param ca3sigma: 注意需要比较大（>=5），否则训练可能非常缓慢
        :param stimunum: 在气味刺激记忆编码阶段，给老鼠呈现多少个不同的气味
        """
        super(HPCSingleLoop, self).__init__()
        self.ts = ts
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.ca3num = ca3num
        self.ca1bias = nn.Parameter(torch.zeros(ca1num))
        self.ec5bias = nn.Parameter(torch.zeros(ecnum))
        self.wca3ca1 = nn.Parameter(torch.rand(ca3num, ca1num) * 0.01)
        self.wec3ca1 = nn.Parameter(torch.randn(ecnum, ca1num) * 0.01)
        self.wca1ec5 = nn.Parameter(torch.randn(ca1num, ecnum) * 0.01)
        # self.wec5ec3 = nn.Parameter(torch.rand(ecnum)*0.01)       # 因为每个EC3分别对应EC5，因此实际上是单位阵，就不初始化了

        nn.init.xavier_normal_(self.wca1ec5)
        nn.init.xavier_normal_(self.wec3ca1)

    def forward(self, ec3input, ca3, ec3_last, ec5_last, ca1_last):
        """

        :param ec3input: (bs, x, ecnum), 在self.cueLocation处传递给所有EC3的额外输入
        :param ec3_last: (bs, ecnum), 继承自上一轮的信息，用于这一轮的细胞的初始化
        :param ec5_last:
        :param ca1_last: (bs, ca1num)
        :return:
            actCell: (bs, actnum), linear输出的结果，结合crossEntropy训练
            ec3his: (trachlength, ecnum)，本轮的EC3发放率历史，只取第一个细胞进行记录
            ec3:    (bs, ecnum), 本轮最后ec3的发放率，可以作为下一轮epoch EC3的初始值
        """
        bs = ec3input.shape[0]
        assert ec3input.shape[1] == self.ecnum
        ec3 = ec3_last
        ec5 = ec5_last
        ca1 = ca1_last

        # ca1 tuft部分必须限制大小，否则可能导致最终结果过大
        ca1 = torch.relu(
            torch.sigmoid(10 * (torch.matmul(ca3, self.wca3ca1) - 0.5))
            * (1 + 3 * torch.sigmoid(torch.matmul(ec3, self.wec3ca1)))
            - self.ca1bias)
        # 注意这里取消了EC5的Bias，如果需要就只能依靠CA1产生，对训练结果没有负面影响
        ec5 = ec5 + 10 * self.ts * torch.matmul(ca1,
                                                self.wca1ec5) + self.ec5bias
        '''
        不同的EC5发放率截断（递归迭代）方法，导致EC5范围不同，进而导致EC3持续时间不同。注意，根据Magee的结论应该EC5均值应取exp(-ts/tau)=0.96
        注意！！！！！代码到这里时，EC5可能会大于1 ！！！！因此考虑sigmoid之后的最大值，不能考虑sigmoid(1)！！！
        '''
        # ec5 = ec5*0.98
        # ec5 = torch.clip(ec5, 0, 1)
        # ec5 = 0.9*ec5 + 0.1*(self.ts * torch.sigmoid(torch.matmul(ca1, self.wca1ec5) + self.ec5bias))  # 用GRU的方式防止EC5越界，但训练不出来
        ec5 = 0.69 + 0.3 * torch.sigmoid(4 * (ec5 - 0.3))  # 与y=x的交点为0.97，最大值0.99

        # ec3 = ec5 * ec3 + (1-ec5) * ec3input  # EC3根据对应的EC5发放率进行衰减
        ec3 = ec5 * ec3 + 0.6 * ec3input  # EC3根据对应的EC5发放率进行衰减

        # ec3 = torch.clip(ec3, 0, 1)       # 其实不需要这个，机制上已经保证了EC3的发放率在01之间。

        return ec3, ec5, ca1


class HPC_DV(nn.Module):
    def __init__(self, ts=0.1, ecnum=100, ca1num=100, ca3num=100, ca3sigma=5, tracklength=100, actnum=2, stimunum=2,
                 loopnum=2):
        super(HPC_DV, self).__init__()
        self.ts = ts  # 时间步长
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.actnum = actnum
        self.stimunum = stimunum  # 一个trail中有几次刺激
        self.ca3num = ca3num
        self.ca3sigma = ca3sigma
        self.loopnum = loopnum
        self.tracklength = tracklength
        self.loopList = nn.ModuleList()  # 用于存储loop的网络
        self.interLayer = nn.ModuleList()
        for i in range(loopnum):
            self.loopList.append(HPCSingleLoop(ts=ts, ca1num=ca1num, ecnum=ecnum, ca3num=ca3num))
            self.interLayer.append(nn.Linear(self.ca1num, self.ecnum))
        self.wca1act = nn.Parameter(nn.init.xavier_normal(torch.randn(ca1num, actnum)))
        self.actbias = nn.Parameter(torch.zeros(actnum))

    def getCA3output(self, x):
        """

        :param x: 当前的位置
        :return: 每个ca3细胞的发放率
        """
        ca3center = torch.linspace(0, self.tracklength, self.ca3num)
        return torch.exp(-(ca3center - x) ** 2 / self.ca3sigma ** 2 / 2)

    def forward(self, cue_ec3input, ec3_last, ec5_last, ca1_last):
        """

        :param cue_ec3input: (bs, tracklength, ecnum), 在self.cueLocation处传递给所有EC3的额外输入
        :param ec3_last: (bs, ecnum), 继承自上一轮的信息，用于这一轮的细胞的初始化
        :param ec5_last:
        :param ca1_last: (bs, ca1num)
        :return:
            actCell: (bs, actnum), linear输出的结果，结合crossEntropy训练
            ec3his: (loopnum, trachlength, ecnum)，本轮的EC3发放率历史，只取第一个细胞进行记录
            ec3:    (bs, ecnum), 本轮最后ec3的发放率，可以作为下一轮epoch EC3的初始值
        """
        bs = cue_ec3input.shape[0]
        assert cue_ec3input.shape[2] == self.ecnum
        ec3 = ec3_last
        ec5 = ec5_last
        ca1 = ca1_last

        ec3his = torch.zeros(self.loopnum, self.tracklength, self.ecnum)  # 注意，这里只保留bs=0的细胞，以节省存储空间。
        ec5his = torch.zeros(self.loopnum, self.tracklength, self.ecnum)
        ca1his = torch.zeros(self.loopnum, self.tracklength, self.ca1num)

        num_ith_cue = 0  # 目前是在等待第几次刺激
        for x in range(self.tracklength):
            ec3input = cue_ec3input[:, x, :]
            ca3 = self.getCA3output(x)
            for i in range(self.loopnum):
                ec3[i], ec5[i], ca1[i] = self.loopList[i](ec3input, ca3, ec3[i], ec5[i], ca1[i])
                ec3his[i, x, :] = ec3[i][0, :]
                ec5his[i, x, :] = ec5[i][0, :]
                ca1his[i, x, :] = ca1[i][0, :]
                ec3input = self.interLayer[i](ca1[i])

        actCell = torch.matmul(ca1[-1], self.wca1act) + self.actbias
        return actCell, ec3his, ec5his, ca1his


def order_cue_pattern_gen(odornum, ecnum):
    cuePattern = torch.zeros(odornum, ecnum)  # 每个气味对应的ec3情况
    for odorIndex in range(odornum):
        cuePattern[odorIndex, odorIndex * 20:(odorIndex + 1) * 20] = 1  # 每十个EC3编码一个气味
    return cuePattern


def order_cue_pickup(cuePattern, odornum, actnum, bs, stimunum, ecnum):
    cue = torch.zeros(bs, stimunum + actnum, ecnum)  # 准备给予模型的线索刺激，包括记忆阶段的stimunum个和分辨阶段的actnum个
    label = torch.zeros(bs)
    for sample in range(bs):
        cue_candidate = list(range(odornum))
        random.shuffle(cue_candidate)
        cue_candidate = cue_candidate[0:stimunum]  # 只取前stimunum个气味（的index）
        odorA_order = random.randint(0, stimunum - 1)
        while True:  # 让AB两个气味必须不相同，这样才能进行先后的比较
            odorB_order = random.randint(0, stimunum - 1)
            if odorB_order != odorA_order:
                break
        odorA_index = cue_candidate[odorA_order]  # Comp处的两个气味选择，注意是原20个气味中的index
        odorB_index = cue_candidate[odorB_order]

        for stim in range(stimunum):
            cue[sample, stim, :] = cuePattern[cue_candidate[stim], :]
        cue[sample, stimunum, :] = cuePattern[odorA_index, :]
        cue[sample, stimunum + 1, :] = cuePattern[odorB_index, :]
        label[sample] = (odorA_order > odorB_order) + 0.0

    return cue, label


def order_input_gen(cue, stimunum, actnum, cuelocation, tracklength):
    """
    每个时刻直接输入给ec3的刺激
    return:
        (bs, tracklength, ecnum)
    """
    num_ith_cue = 0
    bs = cue.shape[0]
    ecnum = cue.shape[2]
    ec3input = torch.zeros(bs, tracklength, ecnum)
    for x in range(tracklength):
        if num_ith_cue < stimunum + actnum and x == cuelocation[num_ith_cue]:
            ec3input[:, x, :] = cue[:, num_ith_cue, :]
            num_ith_cue += 1
    return ec3input


def trace_cue_pattern_gen(actnum, ecnum):
    """

    :param actnum:
    :param ecnum:
    :return:
        cuePattern: (actnum, ecnum)
    """
    cuePattern = torch.zeros(actnum, ecnum)
    cue_ratio = 0.1  # 每个线索激活多少比例的EC3细胞
    for cueType in range(actnum):
        cuePattern[cueType, :] = (torch.rand(1, ecnum) < cue_ratio)  # 每个细胞是否作为本轮被激活的cue细胞。注意有可能有的细胞几个线索都响应
    return cuePattern


def trace_cue_pickup(cuePattern, bs):
    """

    :param cuePattern:
    :param bs:
    :return:
        cue: (bs, ecnum)
        cueUsed: int
    """
    cueUsed = np.random.randint(0, 2, bs)
    cue = cuePattern[cueUsed, :]
    return cue, cueUsed


def trace_input_gen(cue, tracklength):
    """

    :param cue: (bs, ecnum)
    :param tracklength:
    :return: (bs, tracklength, ecnum)
    """
    bs = cue.shape[0]
    ecnum = cue.shape[1]
    ec3input = torch.zeros(bs, tracklength, ecnum)
    ec3input[:, int(0.1 * tracklength), :] = cue
    return ec3input


def net_train(epochNum=500, bs=30, lr=0.003):
    task = 'trace'  # 需要进行的任务类型，
    # task = 'order'

    start_time = time.time()  # 单位；秒
    # 进行网络训练
    tracklength = 100
    ca1num = 100
    ecnum = 100
    actnum = 2
    if task == 'order':
        loopnum = 1  # 似乎order任务中，只能选择loopnum=1才能训练。
    else:
        loopnum = 2  # trace任务中则可以选择更高的loopnum，能否取得更接近于生物的表征？
        labelhis = np.zeros([epochNum])  # 用于绘制sorting图
    odornum = 2
    ordernum = 2  # 每个trail给予几次（顺序的）线索刺激
    cuelocation = 0.1 * tracklength * (torch.Tensor(range(ordernum + actnum)) + 2)  # 多个刺激位置
    net = HPC_DV(loopnum=loopnum, ca1num=ca1num, ecnum=ecnum, tracklength=tracklength, actnum=actnum,
                 stimunum=ordernum + actnum)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    ec3 = [torch.Tensor(1)] * loopnum  # loopnum, bs, ecnum
    ec5 = [torch.Tensor(1)] * loopnum
    ca1 = [torch.Tensor(1)] * loopnum
    for loop in range(loopnum):
        ec3[loop] = torch.zeros(bs, ecnum)
        ec5[loop] = torch.rand(bs, ecnum)
        ca1[loop] = torch.zeros(bs, ca1num)
    ca1_all_his = torch.zeros(epochNum, loopnum, tracklength, ca1num)
    ec5_all_his = torch.zeros(epochNum, loopnum, tracklength, ecnum)
    ec3_all_his = torch.zeros(epochNum, loopnum, tracklength, ecnum)
    losshis = torch.zeros(epochNum)

    if task == 'order':
        cuePattern = order_cue_pattern_gen(odornum, ecnum)
    else:
        cuePattern = trace_cue_pattern_gen(actnum, ecnum)
    for epoch in range(epochNum):
        if task == 'order':
            # order任务：
            cue, label = order_cue_pickup(cuePattern, odornum, actnum, bs, ordernum, ecnum)
            ec3input = order_input_gen(cue, ordernum, actnum, cuelocation, tracklength)
        else:
            # trace任务：
            cue, label = trace_cue_pickup(cuePattern, bs)
            ec3input = trace_input_gen(cue, tracklength)
            labelhis[epoch] = label[0]

        optimizer.zero_grad()
        pred, ec3_this_his, ec5_this_his, ca1_this_his, = net(ec3input, ec3, ec5, ca1)
        loss = criterion(pred, torch.Tensor(label).long())
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)      # 梯度截断
        optimizer.step()

        for loop in range(loopnum):
            ca1_all_his[epoch, loop, :, :] = ca1_this_his[loop, :, :]  # epoch, loop, x, ecnum
            ec5_all_his[epoch, loop, :, :] = ec5_this_his[loop, :, :]
            ec3_all_his[epoch, loop, :, :] = ec3_this_his[loop, :, :]
            ec3[loop] = ec3[loop].detach()
            ec5[loop] = ec5[loop].detach()
            ca1[loop] = ca1[loop].detach()

        print('%d, %.5f' % (epoch, loss.item()))
        losshis[epoch] = loss
    duration_time = time.time() - start_time
    print('总训练时间：%.2f' % duration_time)
    np.savez('cells.npz', ca1_all_his.data.numpy(), ec5_all_his.data.numpy(), ec3_all_his.data.numpy(), labelhis)
    plt.plot(losshis.data.numpy())
    plt.title('batch size: %d, lr: %.4f' % (bs, lr))
    plt.savefig('./fig_result/_loss result.jpg')
    plt.show()


if __name__ == '__main__':
    epochNum = 300
    batch_size = 30
    lr = 0.003
    cueNum = 2

    mode = 'training'  # 本次运行，是训练还是观察结果
    # mode = 'viewing'

    # ——————————训练网络并保存网络结构、中间运行过程
    if mode == 'training':
        seed = 4
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.makedirs('./fig_result', exist_ok=True)  # 保证目录的存在
        net_train(epochNum, batch_size, lr)

    # ——————————读取网络并测试和查看权重
    if mode == 'viewing':
        D = np.load('cells.npz')
        os.makedirs('./fig_result', exist_ok=True)  # 保证目录的存在
        ca1his, ec5his, ec3his, labelhis = D['arr_0'], D['arr_1'], D['arr_2'], D['arr_3']

        history = ca1his[:, 0, :, :]  # 只选择一类细胞进行绘制，且只选择一个层的loop

        # 绘制sorting图
        for cueType in range(cueNum):
            for epoch in range(epochNum):
                if labelhis[epochNum - epoch - 1] == cueType:  # 寻找到最靠后的、指定cueType的epoch
                    last_ca1_rate = history[epochNum - epoch - 1, :, :]  # x, index
                    if cueType == 0:  # 用cueType=0的细胞的排序，作为之后所有cueType的排序
                        sorted_index_cue_0, is_cell_silent = myplot.sorting_plot(last_ca1_rate, str(cueType))
                    else:
                        myplot.sorting_plot(last_ca1_rate, str(cueType), sorted_index_cue_0,
                                            given_silence=is_cell_silent)
                    break

        # 绘制每个细胞的发放率情况, 并保存为图片
        plt.clf()
        for i in range(100):
            mat = history[:, :, i]  # epoch, x
            plt.title('cell %d, max=%.5f' % (i, np.amax(mat)))
            myplot.heatmap(mat, isShow=False, isSave=('./fig_result/%d.jpg' % i))
            plt.close()
            print('cell fig %d saved. ' % i)

    print('end')
