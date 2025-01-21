"""
用EC3input-P10、P01的方式来连续地描述EC3，希望一举解决Sensory输入与CA1输入合并的问题
24.09.22
from：dv.py

取得了相当不错的效果，并且模型也更贴近生物的机制了
"""
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.lick import cs
from src.tools import myplot, loss_smooth


class HPCSingleLoop(nn.Module):
    def __init__(self, ts=0.1, ecnum=100, ca1num=100, ca3num=100, ca3sigma=5, tracklength=100, is_continue=True, device=None):
        """
        单层的Loop。注意每次的forward是在一个x处运行一次迭代，而不是走完整个tracklength

        :param ts: 时间步长，单位为秒
        :param ecnum: EC3 & EC5的细胞数量相同，一一对应
        :param ca1num:
        :param ca3sigma: 注意需要比较大（>=5），否则训练可能非常缓慢
        :param is_continue: 是否用默认的连续方式进行运算。否则的话，转用Markov的离散模式，模仿EC3进行运算
        """
        super(HPCSingleLoop, self).__init__()
        self.device = device
        self.ts = ts
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.ca3num = ca3num
        self.ca3sigma = ca3sigma
        self.ca3order = np.array(range(ca3num))
        self.shuffleCA3()
        self.tracklength = tracklength
        self.ca1bias = nn.Parameter(torch.zeros(ca1num))
        self.ca1tuftbias = nn.Parameter(torch.zeros(ca1num))
        self.ec5bias = nn.Parameter(torch.zeros(ecnum))
        self.lambda_ec5 = 0.05  # 对EC5施加的正则化的强度
        self.wca3ca1 = nn.Parameter(torch.rand(ca3num, ca1num) / ca3num)
        self.wec3ca1 = nn.Parameter(torch.randn(ecnum, ca1num) * 0.01)
        self.wca1ec5 = nn.Parameter(torch.randn(ca1num, ecnum) * 0.01)
        self.wec5ec3 = nn.Parameter(torch.randn(ecnum, ecnum) * 0.01)  # 没什么卵用，并不会加速训练，直接扔了
        # self.wec5ec3 = torch.eye(ecnum)
        self.bn = nn.BatchNorm1d(ecnum, track_running_stats=False)  # 官方的BN还加上了一个可学习的仿射变换

        nn.init.xavier_normal_(self.wca1ec5)
        nn.init.xavier_normal_(self.wec3ca1)
        nn.init.eye_(self.wec5ec3)  # 当前使用单位阵作为ec5-ec3权重的初值，训练之后的结果其实也比较接近单位阵。但，在泛化时是否能够正常迁移呢？

        self.is_continue = is_continue
        self.multiple_factor = 1

    def shift_to_discrete(self, multiple_factor=1):
        """
        将原本的连续性模型转变为离散Markov模型，每个细胞用若干个离散细胞来替代
        :param multiple_factor: 每一个原连续的EC3指数衰减，需要使用多少个离散的On Off细胞进行仿真
        :return:
        """
        self.is_continue = False
        self.eval()
        self.multiple_factor = multiple_factor

    def shuffleCA3(self, order=None):
        """
        重新排布CA3的顺序，用于模仿空间remap
        :return:
        """
        if order is None:
            np.random.shuffle(self.ca3order)
        else:
            self.ca3order = self.ca3order[order]

    def normpdf(self, x, sigma, centers):
        return torch.exp(-(centers - x) ** 2 / sigma ** 2 / 2)

    def getCA3output(self, x):
        """
        根据当前的ca3排布，输出ca3的发放率; 让CA3的表征是首尾相连的，这样在UMAP的时候效果更像【去相关】文中的。
        :param x: 当前的位置
        :return: 每个ca3细胞的发放率
        """
        ca3center = torch.linspace(0, self.tracklength, self.ca3num)
        ca3center = ca3center[self.ca3order]
        temp = self.normpdf(x, self.ca3sigma, ca3center) + \
               self.normpdf(x + self.tracklength, self.ca3sigma, ca3center) + \
               self.normpdf(x - self.tracklength, self.ca3sigma, ca3center)
        return temp.to(self.device)

    def sigp01(self, t):
        """
        P10在P01之上。
        EC3 On的概率，收敛值为 sigp01 / (sigp01 + sigp10), tau值为 1./(sigp01+sigp10)

        t为input，即Sensory输入 + 上游dorsal CA1输入 + EC5输入;
        输入t较小时，拒绝input写入并保持信息（tau大，stationary小）;
        输入t中等时，拒绝input写入并遗忘（tau小， stationary大）;
        输入t较大时，允许input写入并遗忘 (tau小，stationary大）.
        """
        return 0.001 + 0.8 * torch.sigmoid(4 * (t - 1.5))  # 0.03代表惊喜转换率，0.8代表最大的转换速率，可以大于1

    def sigp10(self, t):
        return 0.02 + 0.6 * torch.sigmoid(10 * (t - 0.5))  # 0.01代表静息最小遗忘率，0.6代表最大的转换速率，可以大于1

    def thresh(self, t):
        # 输入不够强的时候就没有输出
        return t - torch.clip(t, -1, 1)

    def forward(self, ec3input, x, ec3_last, ec5_last, ca1_last):
        """

        :param ec3input: (bs, ecnum), 在self.cueLocation处传递给所有EC3的额外输入
        :param ec3_last: (bs, ecnum), 继承自上一轮的信息，用于这一轮的细胞的初始化; 如果是离散模式的话是(bs, ecnum, multiple_factor)
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

        ca3 = self.getCA3output(x)
        if self.is_continue:  # EC3连续模式
            ca1_tuft_input = torch.matmul(ec3, self.wec3ca1) - self.ca1tuftbias
        else:
            ca1_tuft_input = torch.matmul(torch.mean(ec3.float(), dim=2), self.wec3ca1) - self.ca1tuftbias

        # 如果将basal部分的sigmoid非线性函数取消，则可能导致无法训练；系数是为了提高训练速度
        ca1 = torch.relu(
            torch.relu(torch.matmul(ca3, self.wca3ca1))
            * (0.2 + 1 * (torch.sigmoid(ca1_tuft_input)))
            - self.ca1bias)

        # ec5 = ec5 + self.ts * self.thresh(torch.matmul(ca1, self.wca1ec5) + self.ec5bias)  # thresh用于避免波动；但可能梯度消失
        ec5 = ec5 + self.ts * torch.matmul(ca1, self.wca1ec5) + self.ec5bias
        # ec5 = ec5 - self.lambda_ec5 * (ec5 - 0.)  # 加上这个之后，泛化会变得越来越快
        ec5 = torch.clip(ec5, -1, 1)  # 如果不clip，EC5必将爆炸；范围取-1~1、0~1对结果会造成很大影响

        # 这里使用了sigmoid作为非线性，来让EC3的input处于绝佳位置（默认状态下为0.5，因为wec5ec3的默认权重是高斯的）
        ec3_all_input = torch.sigmoid(torch.matmul(ec5, self.wec5ec3))  # Sigmoid相当于让EC3input的范围为0.26~0.73，训练会非常快
        ec3_all_input = ec3_all_input + ec3input
        # ec3_all_input = torch.matmul(ec5, self.wec5ec3) + 0.5  # 不进行Sigmoid限制的话，很有可能由梯度消失问题（loss很长的平台期）
        # ec3_all_input = 0.5 * ec3_all_input / (torch.std(ec3_all_input, dim=1, keepdim=True)+0.0001)  # 层归一化，但不能加速训练（为啥？
        # ec3_all_input = ec3_all_input - torch.mean(ec3_all_input, dim=1, keepdim=True) + 0.5
        ec3_all_input = ec3_all_input / (torch.std(ec3_all_input, dim=0, keepdim=True) + 0.01)  # 批归一化，能极大加速训练，但eval准确率最多99%
        ec3_all_input = ec3_all_input - torch.mean(ec3_all_input, dim=0, keepdim=True) + 0.5
        # ec3_all_input = self.bn(ec3_all_input)  # 使用官方的批归一化，注意自动就包含了仿射变换（通过BP训练参数）因此不需要+0.5；效果跟我的BN差不多

        # 在归一化之后再加上sensory/dorsal输入吗？

        rate01 = self.sigp01(ec3_all_input) * self.ts  # 等价于dr/dt = (1-r)*sigp01 - r*sigp10
        rate10 = self.sigp10(ec3_all_input) * self.ts
        if self.is_continue:
            # EC3大小为：（bs，ecnum）
            ec3 = (1 - ec3) * rate01 + ec3 * (1 - rate10) + 0.02 * 0.3 * torch.randn_like(ec3)  # 加上了一个噪声，强度和仿真markov的一致
        else:
            # 用完全Markov的方式来处理，ec3大小为：（bs，ecnum, multiple_factor)
            trans01 = torch.rand(bs, self.ecnum, self.multiple_factor).to(self.device) < rate01.unsqueeze(2).expand(bs, self.ecnum, self.multiple_factor)
            trans10 = torch.rand(bs, self.ecnum, self.multiple_factor).to(self.device) < rate10.unsqueeze(2).expand(bs, self.ecnum, self.multiple_factor)
            new_state = ec3
            new_state[torch.logical_and(torch.logical_not(ec3), trans01)] = True
            new_state[torch.logical_and(ec3, trans10)] = False
            ec3 = new_state
        return ec3, ec5, ca1


class HPC_DV(nn.Module):
    def __init__(self, ts=0.1, ecnum=100, ca1num=100, ca3num=1000, ca3sigma=5, tracklength=100, loopnum=2, actnum=2, device=None):
        super(HPC_DV, self).__init__()
        self.ts = ts  # 时间步长
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.ca3num = ca3num
        self.actnum = actnum
        self.ca3sigma = ca3sigma
        self.loopnum = loopnum
        self.tracklength = tracklength
        self.loopList = nn.ModuleList()  # 用于存储loop的网络
        self.interLayer = nn.ModuleList()  # 连接上一层的CA1和下一层的EC3
        for i in range(loopnum):
            self.loopList.append(HPCSingleLoop(ts=ts, ca1num=ca1num, ca3num=ca3num, ecnum=ecnum, tracklength=tracklength, device=device))
            linear = nn.Linear(self.ca1num, self.ecnum)
            torch.nn.init.xavier_normal_(linear.weight)
            self.interLayer.append(linear)
        self.wca1act = nn.Parameter(torch.randn(ca1num, actnum))
        torch.nn.init.xavier_normal_(self.wca1act)
        self.actbias = nn.Parameter(torch.zeros(actnum))
        self.ca3order = None
        self.shuffleCA3()

        self.wca3ca1_his = torch.zeros(510, ca3num, ca1num)
        self.epoch = 0

    def init_wca1act(self):
        self.wca1act = nn.Parameter(torch.randn(self.ca1num, self.actnum) * 0.01)
        nn.init.xavier_normal_(self.wca1act)
        self.actbias = nn.Parameter(torch.zeros(self.actnum))

    def set_wca1act(self, wca1act, actbias):
        self.wca1act = wca1act
        self.actbias = actbias

    def shift_to_discrete(self, multiple_factor=1):
        self.eval()
        for i in range(self.loopnum):
            self.loopList[i].shift_to_discrete(multiple_factor=multiple_factor)

    def shuffleCA3(self):
        self.ca3order = list(range(self.ca3num))  # 对DV轴上所有CA3使用相同的方式进行重新排序
        random.shuffle(self.ca3order)
        for loop in range(self.loopnum):
            self.loopList[loop].shuffleCA3(self.ca3order)

    def get_ca3_order(self):
        return self.ca3order

    def set_ca3_order(self, givenCA3order):
        self.ca3order = givenCA3order
        for loop in range(self.loopnum):
            self.loopList[loop].shuffleCA3(self.ca3order)

    def forward(self, cue_ec3input, ec3_last, ec5_last, ca1_last, isSave=True, isSaveAllBatch=False):
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

        if isSave:
            if isSaveAllBatch:
                ec3his = torch.zeros(bs, self.loopnum, self.tracklength, self.ecnum)  # 注意，这里只保留bs=0的细胞，以节省存储空间。
                ec5his = torch.zeros(bs, self.loopnum, self.tracklength, self.ecnum)
                ca1his = torch.zeros(bs, self.loopnum, self.tracklength, self.ca1num)
            else:
                ec3his = torch.zeros(self.loopnum, self.tracklength, self.ecnum)  # 注意，这里只保留bs=0的细胞，以节省存储空间。
                ec5his = torch.zeros(self.loopnum, self.tracklength, self.ecnum)
                ca1his = torch.zeros(self.loopnum, self.tracklength, self.ca1num)
        else:
            ec3his, ec5his, ca1his = None, None, None
        actlist = torch.zeros(bs, self.tracklength, self.actnum)  # 是float取值而不是01值
        for x in range(self.tracklength):
            ec3input = cue_ec3input[:, x, :]
            for i in range(self.loopnum):
                ec3[i], ec5[i], ca1[i] = self.loopList[i](ec3input, x, ec3[i], ec5[i], ca1[i])
                if isSave:
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
        self.wca3ca1_his[self.epoch, :, :] = self.loopList[0].wca3ca1.detach()
        self.epoch += 1
        return actlist, ec3his, ec5his, ca1his, ec3, ec5, ca1


def net_train(loopnum, task, epochNum=500, bs=30, lr=0.003, cueNum=None, given_net=None, lambda_ec5=0.0, given_cuePattern=None, isShuffleCA3=False,
              givenCA3order=None, given_wca1act=None, given_actbias=None, pre_losshis=None, is_plot_block=True, single_act_point=False, device=torch.device("cpu")):
    """

    :param device: 使用的设备。如果是CPU则直接不用管，GPU则要给出，用于CA3的计算。
    :param single_act_point: 是否任务只需要学习一个位置处的lick行为。如果True，则只使用focus_zone的最后一个位置训练；
        如果False，则需要学习整条轨道上的lick行为；虽然更加真实但是似乎不太利于泛化
    :param is_plot_block: plt绘制是否阻断进程. 默认为阻断
    :param lambda_ec5: 对于 EC5-0.5 的二范数正则，默认值相对来说算是小的，可以加速训练; 如果太大可能训练失败
    :param loopnum: 模型中总共包含了几个loop
    :param task: jiajian, decorrelate, 1234, lap
    :param epochNum:
    :param bs:
    :param lr:
    :param cueNum: 在Sequence任务中使用的。主要是在泛化过程中逐渐增加气味的数量
    :param given_net: 已经训练好的模型，如果有的话（可以从hpc_rnn.pth中加载）
    :param given_cuePattern:
    :param isShuffleCA3: 是否重新排布的CA3的序列，用于进行环境remap泛化实验
    :param pre_losshis: 过去模型的loss表现，如果有的话可以用来放在loss曲线图中进行比较
    :return: losshis, numpy
    """
    # torch.autograd.set_detect_anomaly(True)  # 自动追踪NaN的出现
    start_time = time.time()  # 单位；秒
    plt.close()

    '''超参数'''
    # tracklength = 200 if task == 'jiajian' else 100  # 如果需要观察任务之后信息的缓慢遗忘，就在jiajian中使用长轨道
    tracklength = 100
    ca1num = 100
    ecnum = 100
    actnum = 2

    '''初始化网络'''
    if given_net is None:
        net = HPC_DV(loopnum=loopnum, ca1num=ca1num, ecnum=ecnum, tracklength=tracklength, actnum=actnum, device=device)
    else:
        net = given_net
    if given_wca1act is not None:
        net.wca1act = given_wca1act
        net.actbias = given_actbias
    else:
        net.init_wca1act()  # 关闭这个会导致泛化时的训练速度超级慢，但是在多任务中又应该进行保留，就很矛盾
    if isShuffleCA3:
        net.shuffleCA3()
    if givenCA3order is not None:
        ca3order = givenCA3order
        net.set_ca3_order(givenCA3order)
    else:
        ca3order = net.get_ca3_order()
    net.to(device)  # 转移到GPU
    net.train()

    # 对于有奖励但没舔到时有非常大的惩罚，舔而没有奖励时则惩罚较小
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(
        [1, 1.2 if task == 'sequence' or task == 'evidence' or task == 'evidence_poisson' or task == 'sequence' else 5]))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 初始化细胞
    ec3_last = [torch.Tensor(1)] * loopnum  # loopnum, bs, ecnum
    ec5_last = [torch.Tensor(1)] * loopnum
    ca1_last = [torch.Tensor(1)] * loopnum
    for loop in range(loopnum):
        ec3_last[loop] = torch.zeros(bs, ecnum).to(device)
        ec5_last[loop] = torch.rand(bs, ecnum).to(device)
        ca1_last[loop] = torch.zeros(bs, ca1num).to(device)
    ca1_all_his = torch.zeros(epochNum, loopnum, tracklength, ca1num)
    ec5_all_his = torch.zeros(epochNum, loopnum, tracklength, ecnum)
    ec3_all_his = torch.zeros(epochNum, loopnum, tracklength, ecnum)
    losshis = torch.zeros(epochNum)
    actionhis = torch.zeros(epochNum, bs, tracklength)  # 用于绘制lick历史

    '''初始化cue相关'''
    if cueNum is None:
        cueNum = cs.get_cue_num(task)  # set up cue num if not provided.
    cueUsed = np.random.randint(0, cueNum, bs)  # 初始化一下，用于传递给之后的pickup函数
    if given_cuePattern is None:
        cuePattern = cs.cue_gen_StimuNum(cueNum, ecnum)
    else:
        cuePattern = given_cuePattern
    cueUsedhis = np.zeros([epochNum, bs])

    '''loss计数器'''
    loss_smoother = loss_smooth.LossSmoother(window_width=10)

    '''正式进行epoch迭代'''
    real_epochNum = epochNum
    smoothed_loss = 10000
    for epoch in range(epochNum):
        # 如果平滑后的loss已经很小了，就直接终止训练
        if smoothed_loss < 0.015:
            real_epochNum = epoch
            print('End Training.')
            break
        cueUsed, ec3input, lick_label, cue_left_right, focus_zone = \
            cs.get_anything_by_task(task, bs, ecnum, cuePattern, cueUsed, tracklength)  # 直接获取输入数据
        cueUsedhis[epoch, :] = cueUsed
        optimizer.zero_grad()
        pred, ec3_this_his, ec5_this_his, ca1_this_his, ec3_last, ec5_last, ca1_last \
            = net(ec3input.to(device), ec3_last, ec5_last, ca1_last, isSaveAllBatch=False)
        isLicked = torch.argmax(pred, 2) == 1  # (bs, tracklength)
        actionhis[epoch, :, :] = isLicked

        if single_act_point:
            pred = pred[:, focus_zone[-1], :].reshape(bs, actnum)
            lick_label = lick_label[:, focus_zone[-1]]
        else:
            pred = pred.reshape(bs * tracklength, actnum)
            lick_label = lick_label.reshape(bs * tracklength)
        loss_lick = criterion(pred, torch.Tensor(lick_label).long()).cpu()
        loss_ec5 = torch.zeros(bs, ecnum)  # 对EC5的二范数正则，会导致Dorsal CA1数量稀少，但EC5全都趋于0.5
        for loop in range(loopnum):
            loss_ec5 += ((ec5_last[loop] - 0.) ** 2).cpu()
        loss_ec5 = torch.mean(loss_ec5)
        loss_basal_negative = 0
        for i in range(loopnum):
            loss_basal_negative = loss_basal_negative + torch.sum(torch.relu(-net.loopList[i].wca3ca1[net.loopList[i].wca3ca1 < 0.001]))
        # loss = loss_lick + lambda_ec5 * loss_ec5 + loss_basal_negative  # 加上对于CA3->CA1的负权重的惩罚
        loss = loss_lick
        losshis[epoch] = loss_lick
        smoothed_loss = loss_smoother.update_loss(loss_lick.item())  # 经过滑动窗平滑之后的loss结果
        print('%d, loss: %.5f, s_loss: %.3f' % (epoch, loss_lick.item(), smoothed_loss))
        loss.to(device).backward()
        optimizer.step()
        for loop in range(loopnum):
            ca1_all_his[epoch, loop, :, :] = ca1_this_his.cpu()[loop, :,
                                             :]  # epoch, loop, x, cellnum, 只记录batch中第一个sample
            ec5_all_his[epoch, loop, :, :] = ec5_this_his.cpu()[loop, :, :]
            ec3_all_his[epoch, loop, :, :] = ec3_this_his.cpu()[loop, :, :]
            ec3_last[loop] = ec3_last[loop].detach()
            ec5_last[loop] = ec5_last[loop].detach()
            ca1_last[loop] = ca1_last[loop].detach()

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

    if pre_losshis is None:  # train 状态下的处理方式，只绘制本此训练的loss结果
        plt.plot(losshis.data.numpy())
        loss_his_to_save = losshis.data.numpy()
    else:  # Generalization 状态下的处理方式，绘制本轮的loss曲线和之前所有训练中的loss曲线
        if isinstance(pre_losshis, dict):
            trained_num = len(pre_losshis)
        else:  # 之前从未将losshis保存为字典的话，就创建字典
            pre_losshis = {'0': pre_losshis}
            trained_num = 1
        pre_losshis[str(trained_num)] = losshis.data.numpy()  # 增加这一次的loss 记录
        loss_his_to_save = pre_losshis
        for key, trained_loss in pre_losshis.items():
            if key == 0:
                plt.plot(trained_loss, label=key, color='black')  # 加上以前的losshis的曲线，用于比较loss下降速度
            else:
                plt.plot(trained_loss, label=key)
        plt.legend()
    torch.save({'task': task, 'cuePattern': cuePattern, 'net': net, 'losshis': loss_his_to_save, 'ca3order': ca3order,
                'wca1act': net.wca1act, 'actbias': net.actbias}, 'hpc_rnn.pth')
    # plt.title('batch size: %d, lr: %.4f' % (bs, lr))
    plt.grid()
    plt.ylim(0, 1)
    plt.draw()
    plt.savefig('./fig_result/_loss result.jpg')
    if is_plot_block:
        plt.show()

    '''对模型进行测试，获得任务准确率，并将整个batch的结果保存'''
    with torch.no_grad():
        net.eval()
        eval_bs = 1000  # 用于eval的样本数量，越大越好
        last_cueUsed, ec3_last, ec5_last, ca1_last = \
            cs.eval_prepare(task, cueNum, bs, ec3_last, ec5_last, ca1_last, cueUsed, eval_bs=eval_bs, loopnum=loopnum)
        cueUsed_eval, ec3input_eval, lick_label_eval, cue_left_right, focus_zone = \
            cs.get_anything_by_task(task, eval_bs, ecnum, cuePattern, last_cueUsed, tracklength)  # 直接获取输入数据
        pred, ec3_last_his, ec5_last_his, ca1_last_his, _, _, _ = net(ec3input_eval.to(device),
                                                                      ec3_last, ec5_last, ca1_last, isSaveAllBatch=True)
        if single_act_point:
            pred = pred[:, focus_zone[-1], :].reshape(eval_bs, actnum)  # 只关注focus_zone区域最后一个位置的准确率
            label_test = lick_label_eval[:, focus_zone[-1]]
            acc = torch.mean(((torch.argmax(pred, 1).reshape(eval_bs) == label_test) + 0).float())
        else:
            pred = pred[:, focus_zone, :].reshape(eval_bs * len(focus_zone), actnum)  # 只关注focus_zone区域部分的准确率
            label_test = lick_label_eval[:, focus_zone].reshape(eval_bs * len(focus_zone))
            acc = torch.mean(((torch.argmax(pred, 1).reshape(eval_bs * len(focus_zone)) == label_test) + 0).float())
        print('accuracy: %.5f' % acc.cpu().float())
        np.savez('eval_result.npz', ca1_last_his.cpu().data.numpy(),
                 ec5_last_his.cpu().data.numpy(), ec3_last_his.cpu().data.numpy(),
                 cueUsed_eval, cue_left_right)  # (eval_bs, loopnum, tracklength, cell); cueUsed:(eval_bs)
        print('evaluation done.')
    return losshis.data.numpy(), net


def training(loopnum, task, epochNum, batch_size, lr, is_plot_block=True, single_act_point=False, device=torch.device("cpu")):
    return net_train(loopnum, task, epochNum, batch_size, lr, is_plot_block=is_plot_block, single_act_point=single_act_point, device=device)


def viewing(task, cueNum, loop_to_plot=0, cell_type_plot='ca1', is_plot_each_cell=True):
    D = np.load('cells.npz')
    # ca1his: (epoch, loop, x, cell); cueUsedhis: (epoch, bs); actionhis: (epoch, bs)
    ca1his, ec5his, ec3his, cueUsedhis, actionhis = D['arr_0'], D['arr_1'], D['arr_2'], D['arr_3'], D['arr_4']
    Deval = np.load('eval_result.npz', allow_pickle=True)  # 因为这里的cue_left_right有可能是None，因此需要允许pickle
    ca1_last_his, ec5_last_his, ec3_last_his, cueUsed_eval, cue_left_right = Deval['arr_0'], Deval['arr_1'], Deval[
        'arr_2'], Deval[
        'arr_3'], Deval['arr_4']

    if cell_type_plot == 'ec3':
        train_his = ec3his[:, loop_to_plot, :, :]  # 只选择一类细胞、一个loop进行绘制
        eval_his = ec3_last_his[:, loop_to_plot, :, :]  # 选择是第几层loop
    elif cell_type_plot == 'ec5':
        train_his = ec5his[:, loop_to_plot, :, :]  # 只选择一类细胞、一个loop进行绘制
        eval_his = ec5_last_his[:, loop_to_plot, :, :]  # 选择是第几层loop
    else:
        train_his = ca1his[:, loop_to_plot, :, :]
        eval_his = ca1_last_his[:, loop_to_plot, :, :]  # 选择是第几层loop

    # 如果启用MDS的话，可能速度很慢，毕竟维度(100*100)很高
    # 如果需要绘制Splitness曲线，必须要将 is_silence_detect=False 加进去！
    myplot.viewing(task, cueNum, train_his, actionhis, cueUsedhis, eval_his, cueUsed_eval, cue_left_right,
                   is_plot_each_cell=is_plot_each_cell, is_silence_detect=False, isMDS=False)


def generalizing(loopnum, task, epochNum, batch_size, lr, cueNum=None, given_ca3_order=None, given_wca1act=None, given_actbias=None,
                 gen_type='change_ec3', hpc_rnn_path='hpc_rnn.pth', is_plot_block=True, single_act_point=False, device=torch.device("cpu")):
    temp = torch.load(hpc_rnn_path)
    net = temp['net']
    losshis = temp['losshis']  # 注意，如果是刚刚training完，则是一个数组；如果已经进行过泛化，则是一个字典
    cuePattern = temp['cuePattern']
    # assert (temp['task'] == task)  # 这里之所以不再进行assert，是因为要测试不同任务之间泛化时的表现
    if temp['task'] != task:
        warnings.warn('task of generalize has changed. ')

    lossh = None
    '''不改变Sensory输入，但对于CA3位置进行remap的泛化检验'''
    if gen_type == 'change_ca3':
        lossh, _ = net_train(loopnum, task, epochNum, batch_size, lr, cueNum=cueNum, given_net=net, given_cuePattern=cuePattern,
                          isShuffleCA3=True, givenCA3order=given_ca3_order, given_wca1act=None, given_actbias=None,
                          pre_losshis=losshis, is_plot_block=is_plot_block, single_act_point=single_act_point, device=device)
    '''改变Sensory输入，但CA3的位置不变进行remap的泛化检验'''
    if gen_type == 'change_ec3':
        lossh, _ = net_train(loopnum, task, epochNum, batch_size, lr, cueNum=cueNum, given_net=net, pre_losshis=losshis, givenCA3order=given_ca3_order,
                          given_wca1act=None, given_actbias=None, is_plot_block=is_plot_block, single_act_point=single_act_point, device=device)
    '''什么都不变；用在Sequence任务中，以增加气味数量'''
    if gen_type == 'change_nothing':
        if cueNum is not None:
            if cueNum > cuePattern.shape[0]:  # 如果cueNum变多了，则应当增加一下cuePattern
                cuePattern = torch.vstack((cuePattern, cs.cue_gen_StimuNum(actnum=1)))
            assert(cuePattern.shape[0] == cueNum)
        lossh, _ = net_train(loopnum, task, epochNum, batch_size, lr, cueNum=cueNum, given_net=net,
                          given_cuePattern=cuePattern, givenCA3order=given_ca3_order, given_wca1act=None, given_actbias=None,
                          pre_losshis=losshis, is_plot_block=is_plot_block, single_act_point=single_act_point, device=device)
    return lossh


def discrete(loopnum, task, cueNum, multiple_factor=10, hpc_rnn_path='hpc_rnn.pth', eval_bs=1000, device=torch.device("cpu")):
    """

    :param multiple_factor: 一个连续细胞用几个离散Markov细胞替代
    :param hpc_rnn_path:
    :param eval_bs: 用于eval的样本数量，越大越好
    :return:
    """
    temp = torch.load(hpc_rnn_path)
    net = temp['net']
    net.to(device)
    cuePattern = temp['cuePattern']
    ecnum = net.ecnum
    ca1num = net.ca1num
    actnum = net.actnum
    tracklength = net.tracklength
    net.shift_to_discrete(multiple_factor=multiple_factor)

    last_cueUsed = np.random.randint(0, cueNum, eval_bs)
    ec3_last = [torch.Tensor(1)] * loopnum  # loopnum, bs, ecnum————注意，这里使用的细胞初值都是0，如果用理想值的话准确率还能更高
    ec5_last = [torch.Tensor(1)] * loopnum
    ca1_last = [torch.Tensor(1)] * loopnum
    cueUsed_eval, ec3input_eval, lick_label_eval, cue_left_right, focus_zone = \
        cs.get_anything_by_task(task, eval_bs, ecnum, cuePattern, last_cueUsed, tracklength)  # 直接获取输入数据
    for loop in range(loopnum):
        ec3_last[loop] = torch.zeros(eval_bs, ecnum, multiple_factor).bool().to(device)
        ec5_last[loop] = torch.zeros(eval_bs, ecnum).to(device)
        ca1_last[loop] = torch.zeros(eval_bs, ca1num).to(device)

    # 此时的his变量是None，因为his的处理实在太麻烦了，细胞数量与连续模型的并不一致
    pred, _, _, _, _, _, _ = net(ec3input_eval.to(device), ec3_last, ec5_last, ca1_last,
                                                                  isSave=False)
    pred = pred[:, focus_zone, :].reshape(eval_bs * len(focus_zone), actnum)  # 只关注focus_zone区域部分的准确率
    label_test = lick_label_eval[:, focus_zone].reshape(eval_bs * len(focus_zone))
    acc = torch.mean(((torch.argmax(pred, 1).reshape(eval_bs * len(focus_zone)) == label_test) + 0).float())
    print('accuracy: %.5f' % acc.float())
    print('evaluation done.')


if __name__ == '__main__':
    mode = 'training'  # 本次运行，是训练还是观察结果
    # mode = 'viewing'
    # mode = 'generalizing'  # 必须先train然后才能测试泛化
    # mode = 'discrete'  # 用离散的方式来描述EC3，用训练之后的网络来仿真；效果居然出奇的好

    # task = 'jiajian'  # 任务类型, 注意jiajian任务的轨道长度为200而非100
    # task = 'decorrelate'
    # task = 'lap'
    # task = 'alter'  # 其实并不是真正的Alternative任务，而是Lap=2的情况。但是如果考虑到老鼠对自身行为的观测就一样了
    # task = '1234'
    # task = 'evidence'
    # task = 'evidence_poisson'
    task = 'sequence'
    # task = 'envb'
    # task = 'doublex'

    # single_act_point = True  # 是否只学习一个位置处的action，如果是True的话泛化会更快一些
    single_act_point = False  # 如果是false的话更加生物真实一些

    # 检查device情况
    if torch.cuda.is_available():
        isGPU = True
        device = torch.device("cuda:0")
    else:
        isGPU = False
        device = torch.device("cpu")
    print('\n  -----------Device using: ---------')
    print(device)

    loopnum = 3
    epochNum = 500
    batch_size = 32 if isGPU else 30  # 注意，这里的bs可能会很大程度上影响结果
    if task == 'lap' or task == 'sequence':
        batch_size = 128

    lr = 0.01
    cueNum = cs.get_cue_num(task)

    seed = 2  # 设置全局的随机种子，用来确保实验可复现; lap 500的种子（服务器）
    if task == 'sequence':
        seed = 1
    # seed = int(time.time())  # 保证具有随机性的种子，防止在泛化过程中使用固定的种子
    print(f'\n  ----Seed Using: {seed}')
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ——————————训练网络并保存网络结构、中间运行过程
    if mode == 'training':
        # 对整个轨道上每一个位置lick进行判断
        # net_train(loopnum, task, epochNum, batch_size, lr, is_plot_block=True, single_act_point=single_act_point, device=device)

        # 只针对一个位置上lick进行判断学习
        losshis, net = net_train(loopnum, task, epochNum, batch_size, lr, is_plot_block=True, single_act_point=single_act_point, device=device)
        temp = net.wca3ca1_his[0:losshis.shape[0], :, :]
        torch.save(temp, 'w84.pth')  # 截取有效的epoch

    # ——————————读取网络并测试和查看权重
    if mode == 'viewing':
        loop_to_plot = 1  # 选择是第几个loop进行绘制
        cell_type_plot = 'ca1'  # 选择的是哪种细胞进行绘制
        # cell_type_plot = 'ec3'
        # cell_type_plot = 'ec5'
        viewing(task, cueNum, loop_to_plot, cell_type_plot)

    # ——————————加载模型，打乱CA3并重新训练
    if mode == 'generalizing':
        # 注意，不能多次使用这个函数！因为seed是固定的，已经吃过一次大亏了
        generalizing(loopnum, task, epochNum, batch_size, lr, gen_type='change_nothing', single_act_point=single_act_point, device=device)
        # generalizing(loopnum, task, epochNum, batch_size, lr, gen_type='change_ca3', single_act_point=single_act_point, device=device)

    if mode == 'discrete':
        discrete(loopnum, task, cueNum, multiple_factor=10, device=device)

    print('end')
