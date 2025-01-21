"""
引入多层DV轴结构，并尝试和cs.py代码进行融合，以后可以直接选择dv层数
24.09.16
from：dv_order.py
"""
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.tools import myplot, DecoderCS
from src.lick import cs


class HPCSingleLoop(nn.Module):
    def __init__(self, ts=0.1, ecnum=100, ca1num=100, ca3num=100, ca3sigma=5, tracklength=100):
        """
        单层的Loop。注意每次的forward是在一个x处运行一次迭代，而不是走完整个tracklength

        :param ts: 时间步长，单位为秒
        :param ecnum: EC3 & EC5的细胞数量相同，一一对应
        :param ca1num:
        :param ca3sigma: 注意需要比较大（>=5），否则训练可能非常缓慢
        """
        super(HPCSingleLoop, self).__init__()
        self.ts = ts
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.ca3num = ca3num
        self.ca3sigma = ca3sigma
        self.ca3order = np.array(range(ca3num))
        self.shuffleCA3()
        self.tracklength = tracklength
        self.ca1bias = nn.Parameter(torch.zeros(ca1num))
        self.ec5bias = nn.Parameter(torch.zeros(ecnum))
        self.wca3ca1 = nn.Parameter(torch.rand(ca3num, ca1num) * 0.01)
        self.wec3ca1 = nn.Parameter(torch.randn(ecnum, ca1num) * 0.01)
        self.wca1ec5 = nn.Parameter(torch.randn(ca1num, ecnum) * 0.01)

        nn.init.xavier_normal_(self.wca1ec5)
        nn.init.xavier_normal_(self.wec3ca1)

    def shuffleCA3(self):
        """
        重新排布CA3的顺序，用于模仿空间remap
        :return:
        """
        np.random.shuffle(self.ca3order)

    def getCA3output(self, x):
        """
        根据当前的ca3排布，输出ca3的发放率
        :param x: 当前的位置
        :return: 每个ca3细胞的发放率
        """
        ca3center = torch.linspace(-0.1 * self.tracklength, 1.1 * self.tracklength, self.ca3num)
        ca3center = ca3center[self.ca3order]
        return torch.exp(-(ca3center - x) ** 2 / self.ca3sigma ** 2 / 2)

    def forward(self, ec3input, x, ec3_last, ec5_last, ca1_last):
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

        ca3 = self.getCA3output(x)
        # ca1 tuft部分必须限制大小，否则可能导致最终结果过大
        ca1 = torch.clip(
            torch.sigmoid(10 * (torch.matmul(ca3, self.wca3ca1) - 0.5))
            * (1 + 3 * torch.sigmoid(torch.matmul(ec3, self.wec3ca1)))
            - self.ca1bias, 0, 1)
        # 注意这里取消了EC5的Bias，如果需要就只能依靠CA1产生，对训练结果没有负面影响
        ec5 = ec5 + 10 * self.ts * torch.matmul(ca1, self.wca1ec5) + self.ec5bias
        # 不同的EC5发放率截断（递归迭代）方法，导致EC5范围不同，进而导致EC3持续时间不同。
        # 注意，根据Magee的结论应该EC5均值应取exp(-ts/tau)=0.96
        ec5 = 0.69 + 0.3 * torch.sigmoid(4 * (ec5 - 0.3))  # 与y=x的交点为0.97，最大值0.99

        # ec3 = ec5 * ec3 + 0.6 * ec3input  # EC3根据对应的EC5发放率进行衰减
        # ec3 = ec5 * ec3 + 0.6 * (1 - ec3) * torch.clip(ec3input, -0.5, 0.5)  # 如果不截断EC3的值几乎必定会产生nan
        ec3 = ec5 * ec3 + 0.6 * (1 - ec3) * ec3input  # 如果不截断EC3的值几乎必定会产生nan
        # ec3 = ec5 * ec3 + 0.6 * ec3input  # dv_order中的做法，不知道能不能得到理想的DV表征

        # 加入EC3的随机切变
        is_ec3_noised = (torch.rand(bs, self.ecnum) < 0.04 * self.ts)  # 本轮中每个EC3细胞是否添加noise
        ec3[is_ec3_noised] = 0.5 * ec3[is_ec3_noised] + 0.5 * 0.6  # 用以保证noised之后的EC3发放率在0.6以下
        ec3 = torch.clip(ec3, 0, 1)
        return ec3, ec5, ca1


class HPC_DV(nn.Module):
    def __init__(self, ts=0.1, ecnum=100, ca1num=100, ca3sigma=5, tracklength=100, loopnum=2, actnum=2):
        super(HPC_DV, self).__init__()
        self.ts = ts  # 时间步长
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.actnum = actnum
        self.ca3sigma = ca3sigma
        self.loopnum = loopnum
        self.tracklength = tracklength
        self.loopList = nn.ModuleList()  # 用于存储loop的网络
        self.interLayer = nn.ModuleList()  # 连接上一层的CA1和下一层的EC3
        for i in range(loopnum):
            self.loopList.append(HPCSingleLoop(ts=ts, ca1num=ca1num, ecnum=ecnum, tracklength=tracklength))
            linear = nn.Linear(self.ca1num, self.ecnum)
            torch.nn.init.xavier_normal_(linear.weight)
            self.interLayer.append(linear)
        self.wca1act = nn.Parameter(torch.randn(ca1num, actnum))
        torch.nn.init.xavier_normal_(self.wca1act)
        self.actbias = nn.Parameter(torch.zeros(actnum))

    def init_wca1act(self):
        self.wca1act = nn.Parameter(torch.randn(self.ca1num, self.actnum) * 0.01)
        nn.init.xavier_normal_(self.wca1act)
        self.actbias = nn.Parameter(torch.zeros(self.actnum))

    def forward(self, cue_ec3input, ec3_last, ec5_last, ca1_last, isSaveAllBatch=False):
        """

        :param cue_ec3input: (bs, tracklength, ecnum), 在self.cueLocation处传递给所有EC3的额外输入
        :param ec3_last: (bs, ecnum), 继承自上一轮的信息，用于这一轮的细胞的初始化
        :param ec5_last:
        :param ca1_last: (bs, ca1num)
        :param isSaveAllBatch: 是否将整个batch的history数据进行输出，用于在训练之后对模型进行测试
        :return:
            actList: (bs, tracklength, actnum), linear输出的结果，结合crossEntropy训练
            ec3his: (loopnum, trachlength, ecnum)，本轮的EC3发放率历史，只取第一个细胞进行记录
                如果isSaveAllBatch：(bs, loopnum, tracklength, ecnum)
            ec3: [loopnum](bs, ecnum),
        """
        bs = cue_ec3input.shape[0]
        assert cue_ec3input.shape[2] == self.ecnum
        ec3 = ec3_last
        ec5 = ec5_last
        ca1 = ca1_last

        if isSaveAllBatch:
            ec3his = torch.zeros(bs, self.loopnum, self.tracklength, self.ecnum)  # 注意，这里只保留bs=0的细胞，以节省存储空间。
            ec5his = torch.zeros(bs, self.loopnum, self.tracklength, self.ecnum)
            ca1his = torch.zeros(bs, self.loopnum, self.tracklength, self.ca1num)
        else:
            ec3his = torch.zeros(self.loopnum, self.tracklength, self.ecnum)  # 注意，这里只保留bs=0的细胞，以节省存储空间。
            ec5his = torch.zeros(self.loopnum, self.tracklength, self.ecnum)
            ca1his = torch.zeros(self.loopnum, self.tracklength, self.ca1num)

        actlist = torch.zeros(bs, self.tracklength, self.actnum)  # 是float取值而不是01值
        for x in range(self.tracklength):
            ec3input = cue_ec3input[:, x, :]
            for i in range(self.loopnum):
                ec3[i], ec5[i], ca1[i] = self.loopList[i](ec3input, x, ec3[i], ec5[i], ca1[i])
                if isSaveAllBatch:
                    ec3his[:, i, x, :] = ec3[i][:, :]
                    ec5his[:, i, x, :] = ec5[i][:, :]
                    ca1his[:, i, x, :] = ca1[i][:, :]
                else:
                    ec3his[i, x, :] = ec3[i][0, :]  # 只取第一个细胞进行记录
                    ec5his[i, x, :] = ec5[i][0, :]
                    ca1his[i, x, :] = ca1[i][0, :]
                ec3input = self.interLayer[i](ca1[i])
            actCell = torch.matmul(ca1[-1], self.wca1act) + self.actbias
            actlist[:, x, :] = actCell
        return actlist, ec3his, ec5his, ca1his, ec3, ec5, ca1


def net_train(loopnum, task, epochNum=500, bs=30, lr=0.003, given_net=None, given_cuePattern=None, isShuffleCA3=False,
              pre_losshis=None):
    """

    :param loopnum: 模型中总共包含了几个loop
    :param task: jiajian, decorrelate, 1234, lap
    :param epochNum:
    :param bs:
    :param lr:
    :param given_net: 已经训练好的模型，如果有的话（可以从hpc_rnn.pth中加载）
    :param given_cuePattern:
    :param isShuffleCA3: 是否重新排布的CA3的序列，用于进行环境remap泛化实验
    :param pre_losshis: 过去模型的loss表现，如果有的话可以用来放在loss曲线图中进行比较
    :return:
    """
    # torch.autograd.set_detect_anomaly(True)  # 自动追踪NaN的出现
    start_time = time.time()  # 单位；秒
    '''超参数'''
    tracklength = 100
    ca1num = 100
    ecnum = 100
    actnum = 2

    '''初始化网络'''
    if given_net is None:
        net = HPC_DV(loopnum=loopnum, ca1num=ca1num, ecnum=ecnum, tracklength=tracklength, actnum=actnum)
    else:
        net = given_net
        net.init_wca1act()
    if isShuffleCA3:
        net.shuffleCA3()
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 5]))  # 对于有奖励但没舔到时有非常大的惩罚，舔而没有奖励时则惩罚较小
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 初始化细胞
    ec3_last = [torch.Tensor(1)] * loopnum  # loopnum, bs, ecnum
    ec5_last = [torch.Tensor(1)] * loopnum
    ca1_last = [torch.Tensor(1)] * loopnum
    for loop in range(loopnum):
        ec3_last[loop] = torch.zeros(bs, ecnum)
        ec5_last[loop] = torch.rand(bs, ecnum)
        ca1_last[loop] = torch.zeros(bs, ca1num)
    ca1_all_his = torch.zeros(epochNum, loopnum, tracklength, ca1num)
    ec5_all_his = torch.zeros(epochNum, loopnum, tracklength, ecnum)
    ec3_all_his = torch.zeros(epochNum, loopnum, tracklength, ecnum)
    losshis = torch.zeros(epochNum)
    actionhis = torch.zeros(epochNum, bs, tracklength)  # 用于绘制lick历史

    '''初始化cue相关'''
    if task == 'lap' or task == '1234':
        cueNum = 4
    else:
        cueNum = 2
    cueUsed = np.random.randint(0, cueNum, bs)  # 初始化一下，用于传递给之后的pickup函数
    if given_cuePattern is None:
        cuePattern = cs.cue_gen_StimuNum(cueNum, ecnum)
    else:
        cuePattern = given_cuePattern
    cueUsedhis = np.zeros([epochNum, bs])

    '''正式进行epoch迭代'''
    real_epochNum = epochNum
    loss = torch.Tensor([10000])
    for epoch in range(epochNum):
        # 如果loss已经很小了，就直接终止训练
        if loss.item() < 0.001:
            real_epochNum = epoch
            print('End Training.')
            break
        # 根据任务类型来确定reward位置
        cueUsed, lick_label = cs.get_cueUsed_label_by_task(task, cueUsed, tracklength)
        cueUsedhis[epoch, :] = cueUsed
        if task == 'lap':
            cue = torch.Tensor(np.tile(cuePattern[0, :], [bs, 1]))
        else:
            cue = cuePattern[cueUsed, :]  # 注意，这个操作之后似乎cue变量就变成了0和1的矩阵，需要用*.bool()还原回布尔变量
        ec3input = cs.oneCue_ec3_input_gen(cue, tracklength)

        optimizer.zero_grad()
        pred, ec3_this_his, ec5_this_his, ca1_this_his, ec3_last, ec5_last, ca1_last \
            = net(ec3input, ec3_last, ec5_last, ca1_last, isSaveAllBatch=False)
        isLicked = torch.argmax(pred, 2) == 1  # (bs, tracklength)
        actionhis[epoch, :, :] = isLicked

        pred = pred.reshape(bs * tracklength, actnum)
        lick_label = lick_label.reshape(bs * tracklength)
        loss = criterion(pred, torch.Tensor(lick_label).long())
        loss.backward()
        optimizer.step()

        for loop in range(loopnum):
            ca1_all_his[epoch, loop, :, :] = ca1_this_his[loop, :, :]  # epoch, loop, x, cellnum, 只记录batch中第一个sample
            ec5_all_his[epoch, loop, :, :] = ec5_this_his[loop, :, :]
            ec3_all_his[epoch, loop, :, :] = ec3_this_his[loop, :, :]
            ec3_last[loop] = ec3_last[loop].detach()
            ec5_last[loop] = ec5_last[loop].detach()
            ca1_last[loop] = ca1_last[loop].detach()

        print('%d,  %.5f' % (epoch, loss.item()))
        losshis[epoch] = loss

    # 进行epochNum的截取
    epochNum = real_epochNum
    losshis = losshis[0:epochNum]
    ec3_all_his = ec3_all_his[0:epochNum, :, :, :]
    ec5_all_his = ec5_all_his[0:epochNum, :, :, :]
    ca1_all_his = ca1_all_his[0:epochNum, :, :, :]
    cueUsedhis = cueUsedhis[0:epochNum, :]
    actionhis = actionhis[0:epochNum, :, :]
    # 进行epochNum的截取
    epochNum = real_epochNum
    losshis = losshis[0:epochNum]
    ec3_all_his = ec3_all_his[0:epochNum, :, :, :]
    ec5_all_his = ec5_all_his[0:epochNum, :, :, :]
    ca1_all_his = ca1_all_his[0:epochNum, :, :, :]
    cueUsedhis = cueUsedhis[0:epochNum, :]
    actionhis = actionhis[0:epochNum, :, :]
    duration_time = time.time() - start_time
    print('总训练时间：%.2f' % duration_time)
    np.savez('cells.npz', ca1_all_his.data.numpy(), ec5_all_his.data.numpy(),
             ec3_all_his.data.numpy(), cueUsedhis, actionhis)  # (epochnum, loopnum, x, cell); cueUsedhis:(epochnum, bs)
    plt.plot(losshis.data.numpy())
    if pre_losshis is not None:  # 加上以前的losshis的曲线（黑色），用于比较loss下降速度
        plt.plot(pre_losshis, color='black')
    plt.title('batch size: %d, lr: %.4f' % (bs, lr))
    plt.savefig('./fig_result/_loss result.jpg')
    plt.show()

    '''对模型进行测试，获得任务准确率，并将整个batch的结果保存'''
    net.eval()
    eval_bs = 1000  # 用于eval的样本数量，越大越好
    if task == 'lap':  # 单独准备Lap任务的初始值，因为每轮和每轮之间的cueUsed必须是连续的
        first_index = [-1] * 4  # 搜索四个lap情况作为eval的细胞初值
        for i in range(4):
            for index in range(bs):
                if cueUsed[index] == i:
                    first_index[i] = index
                    break
        assert all(x >= 0 for x in first_index)
        temp = int(eval_bs / 4)  # 复制每一行的次数
        for loop in range(loopnum):
            ec3_last[loop] = np.repeat(ec3_last[loop][first_index, :], repeats=temp,
                                       axis=0)  # 复制每一行并堆叠成(eval_bs, cellNum)的形式
            ec5_last[loop] = np.repeat(ec5_last[loop][first_index, :], repeats=temp, axis=0)
            ca1_last[loop] = np.repeat(ca1_last[loop][first_index, :], repeats=temp, axis=0)
        last_cueUsed = np.repeat(np.array([0, 1, 2, 3]), repeats=temp)
    else:
        last_cueUsed = np.random.randint(0, cueNum, eval_bs)
        for loop in range(loopnum):
            ec3_last[loop] = ec3_last[loop][0, :].repeat(eval_bs, 1)  # 以第一个元素的各个细胞的值，作为eval的初值
            ec5_last[loop] = ec5_last[loop][0, :].repeat(eval_bs, 1)
            ca1_last[loop] = ca1_last[loop][0, :].repeat(eval_bs, 1)

    cueUsed, lick_label_eval = cs.get_cueUsed_label_by_task(task, last_cueUsed, tracklength)
    if task == 'lap':
        cue = torch.Tensor(np.tile(cuePattern[0, :], [eval_bs, 1]))
    else:
        cue = cuePattern[cueUsed, :]  # 注意，这个操作之后似乎cue变量就变成了0和1的矩阵，需要用*.bool()还原回布尔变量
    ec3input = cs.oneCue_ec3_input_gen(cue, tracklength)
    pred, ec3_last_his, ec5_last_his, ca1_last_his, _, _, _ = net(ec3input, ec3_last, ec5_last, ca1_last,
                                                                  isSaveAllBatch=True)
    pred = pred.reshape(eval_bs * tracklength, actnum)
    label_test = lick_label_eval.reshape(eval_bs * tracklength)
    acc = torch.mean(((torch.argmax(pred, 1).reshape(eval_bs * tracklength) == label_test) + 0).float())
    print('accuracy: %.5f' % acc.float())
    np.savez('eval_result.npz', ca1_last_his.data.numpy(), ec5_last_his.data.numpy(), ec3_last_his.data.numpy(),
             cueUsed)  # (eval_bs, loopnum, tracklength, cell); cueUsed:(eval_bs)
    print('evaluation done.')


if __name__ == '__main__':
    # mode = 'training'  # 本次运行，是训练还是观察结果
    mode = 'viewing'
    # mode = 'generalizing'  # 必须先train然后才能测试泛化

    # task = 'jiajian'  # 任务类型
    # task = 'decorrelate'
    task = 'lap'
    # task = '1234'

    loopnum = 2
    epochNum = 500
    batch_size = 30
    lr = 0.003
    if task == 'lap' or task == '1234':
        cueNum = 4
    else:
        cueNum = 2

    seed = 5
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs('./fig_result', exist_ok=True)  # 保证目录的存在

    # ——————————训练网络并保存网络结构、中间运行过程
    if mode == 'training':
        net_train(loopnum, task, epochNum, batch_size, lr)

    # ——————————读取网络并测试和查看权重
    if mode == 'viewing':
        loop_to_plot = 1  # 选择是第几个loop进行绘制

        D = np.load('cells.npz')
        # ca1his: (epoch, loop, x, cell); cueUsedhis: (epoch, bs); actionhis: (epoch, bs)
        ca1his, ec5his, ec3his, cueUsedhis, actionhis = D['arr_0'], D['arr_1'], D['arr_2'], D['arr_3'], D['arr_4']
        train_his = ec3his[:, loop_to_plot, :, :]  # 只选择一类细胞、一个loop进行绘制

        Deval = np.load('eval_result.npz')
        ca1_last_his, ec5_last_his, ec3_last_his, cueUsed_eval = Deval['arr_0'], Deval['arr_1'], Deval['arr_2'], Deval[
            'arr_3']
        eval_his = ec3_last_his[:, loop_to_plot, :, :]  # 选择是第几层loop

        myplot.viewing(task, cueNum, train_his, actionhis, cueUsedhis, eval_his, cueUsed_eval)

    # ——————————加载模型，打乱CA3并重新训练
    if mode == 'generalizing':
        temp = torch.load('hpc_rnn.pth')
        net = temp['net']
        losshis = temp['losshis']
        cuePattern = temp['cuePattern']
        assert (temp['task'] == task)

        '''不改变Sensory输入，但对于CA3位置进行remap的泛化检验'''
        # net_train(task, epochNum, batch_size, lr, given_net=net, given_cuePattern=cuePattern, isShuffleCA3=True,
        #           pre_losshis=losshis)
        '''改变Sensory输入，但CA3的位置不变进行remap的泛化检验'''
        net_train(task, epochNum, batch_size, lr, given_net=net, pre_losshis=losshis)
    print('end')
