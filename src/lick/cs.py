"""
要让小鼠学会在哪个位置进行lick，以及是否进行lick
也可以通过调整loss权重的方式，用最大化Reward的方式，让老鼠学会是否lick

实验效果：
PF更加符合实际一点点，但是还是0~0.9之间没有明显的位置场。
并且！lick行为跟老鼠的行为一模一样！！ oneNote 240911
24.09.10
"""
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.tools import myplot, DecoderCS
from src.lick.evidence_settings import evidence_stimu_num, evidence_stimu_duration
import warnings


class HPCrnn(nn.Module):
    def __init__(self, ts=0.1, ecnum=100, ca1num=100, ca3num=100, ca3sigma=5, tracklength=100, actnum=2):
        """
        :param ts: 时间步长，单位为秒
        :param ecnum: EC3 & EC5的细胞数量相同，一一对应
        :param ca1num:
        :param actnum: 模型能够采取的行动数量
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
        self.ca3order = np.array(range(ca3num))
        self.shuffleCA3()

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

    def shuffleCA3(self):
        """
        重新排布CA3的顺序，用于模仿空间remap
        :return:
        """
        np.random.shuffle(self.ca3order)

    def init_wca1act(self):
        self.wca1act = nn.Parameter(torch.randn(self.ca1num, self.actnum) * 0.01)
        nn.init.xavier_normal_(self.wca1act)
        self.actbias = nn.Parameter(torch.zeros(self.actnum))

    def getCA3output(self, x):
        """
        根据当前的ca3排布，输出ca3的发放率
        :param x: 当前的位置
        :return: 每个ca3细胞的发放率
        """
        ca3center = torch.linspace(-0.1 * self.tracklength, 1.1 * self.tracklength, self.ca3num)
        ca3center = ca3center[self.ca3order]
        return torch.exp(-(ca3center - x) ** 2 / self.ca3sigma ** 2 / 2)

    def forward(self, ec3input, ec3_last, ec5_last, ca1_last, isSaveAll=False):
        """

        :param ec3input: (bs, tracklength, ecnum), 传递给所有EC3的额外输入
        :param ec3_last: (bs, ecnum), 继承自上一轮的信息，用于这一轮的细胞的初始化
        :param ec5_last:
        :param ca1_last: (bs, ca1num)
        :param isSaveAll: 是否将整个batch的history数据进行输出，用于在训练之后对模型进行测试
        :return:
            actlist: (bs, tracklength, actnum), linear输出的结果，结合crossEntropy训练
            ec3his: (trachlength, ecnum)，本轮的EC3发放率历史，只取第一个细胞进行记录
            ec3:    (bs, ecnum), 本轮最后ec3的发放率，可以作为下一轮epoch EC3的初始值
        """
        bs = ec3input.shape[0]
        assert ec3input.shape[2] == self.ecnum
        ec3 = ec3_last
        ec5 = ec5_last
        ca1 = ca1_last
        if isSaveAll:
            ec3his = torch.zeros(bs, self.tracklength, self.ecnum)  # 注意，这里只保留bs=0的细胞，以节省存储空间。
            ec5his = torch.zeros(bs, self.tracklength, self.ecnum)
            ca1his = torch.zeros(bs, self.tracklength, self.ca1num)
        else:
            ec3his = torch.zeros(self.tracklength, self.ecnum)  # 注意，这里只保留bs=0的细胞，以节省存储空间。
            ec5his = torch.zeros(self.tracklength, self.ecnum)
            ca1his = torch.zeros(self.tracklength, self.ca1num)
        actlist = torch.zeros(bs, self.tracklength, self.actnum)  # 是float取值而不是01值
        for x in range(self.tracklength):

            ca3 = self.getCA3output(x)

            # # ca1 tuft部分必须限制大小，否则可能导致最终结果过大; Basal部分的输入是0~1之间，Tuft部分的增益是1~2之间
            # ca1 = torch.relu(
            #     torch.matmul(ca3, self.wca3ca1)
            #     * (1 + torch.sigmoid(torch.matmul(ec3, self.wec3ca1)))
            #     - self.ca1bias)
            ca1 = torch.clip(
                torch.matmul(ca3, self.wca3ca1)
                * (1 + torch.sigmoid(torch.matmul(ec3, self.wec3ca1)))
                - self.ca1bias, 0, 1)

            # 注意这里取消了EC5的Bias，如果需要就只能依靠CA1产生，对训练结果没有负面影响
            ec5 = ec5 + 10 * self.ts * torch.matmul(ca1, self.wca1ec5)

            # 不同的EC5发放率截断（递归迭代）方法，导致EC5范围不同，进而导致EC3持续时间不同。
            # 注意，根据Magee的结论应该EC5均值应取exp(-ts/tau)=0.96
            ec5 = 0.69 + 0.3 * torch.sigmoid(4 * (ec5 - 0.3))  # 与y=x的交点为0.97，最大值0.99

            # # 最经典的做法，没有输入的话就保持不变，有输入则按照0.6的比例上拉到1
            # ec3 = ec5 * ec3
            # cue = (ec3input[:, x, :] > 0)
            # ec3[cue] = (1 - 0.6) * ec3[cue] + 0.6 * 1  # 希望刻画的是，将state=0的cue EC3，以0.6的概率转化为state=1

            # 输入为，sensory*(上拉后的值-ec3)
            ec3 = ec5 * ec3 + 0.6 * (1 - ec3) * ec3input[:, x, :]

            # 加入EC3的随机切变
            is_ec3_noised = (torch.rand(bs, self.ecnum) < 0.04 * self.ts)  # 本轮中每个EC3细胞是否添加noise
            ec3[is_ec3_noised] = 0.5 * ec3[is_ec3_noised] + 0.5 * 0.6  # 用以保证noised之后的EC3发放率在0.6以下
            # ec3 = torch.clip(ec3, 0, 1)       # 其实不需要这个，机制上已经保证了EC3的发放率在01之间。
            actCell = torch.matmul(ca1, self.wca1act) + self.actbias
            actlist[:, x, :] = actCell
            if isSaveAll:
                ec3his[:, x, :] = ec3[:, :]  # 只取第一个细胞进行记录
                ec5his[:, x, :] = ec5[:, :]
                ca1his[:, x, :] = ca1[:, :]
            else:
                ec3his[x, :] = ec3[0, :]  # 只取第一个细胞进行记录
                ec5his[x, :] = ec5[0, :]
                ca1his[x, :] = ca1[0, :]

        return actlist, ec3his, ec5his, ca1his, ec3, ec5, ca1


def cue_gen_StimuNum(actnum, ecnum=100, cue_ratio=0.2):
    """
    根据刺激类型的数量，生成几种刺激编码。用于CS+-任务，CS1234任务和Lap任务
    :param cue_ratio: 每个线索激活多少比例的EC3细胞
    :param actnum: 总共包含几种刺激
    :param ecnum:
    :return:
    """
    cuePattern = torch.zeros(actnum, ecnum)
    for cueType in range(actnum):
        cuePattern[cueType, :] = (torch.rand(1, ecnum) < cue_ratio)  # 每个细胞是否作为本轮被激活的cue细胞。注意有可能有的细胞几个线索都响应
    return cuePattern


jiajian_action_area = range(80, 90)  # 加减任务的action区域


def pickup_jiajian(last_cueUsed, tracklength):
    """
    对于每一个trail，根据线索类型决定每个位置上是否有奖励
    对于CS+、CS-任务而言，只有CS+（cueUsed=0)的时候才需要在一个区间内lick

    :param last_cueUsed: (bs)
    :param tracklength: 注意，这里使用的tracklength推荐为200，也因此修改了lick_label
    :return:
        cueUsed: (bs)
    """
    bs = last_cueUsed.shape[0]
    cueUsed = np.random.randint(0, 2, bs)
    lick_label = torch.zeros(bs, tracklength)
    for i in range(bs):
        if cueUsed[i] == 0:
            lick_label[i, jiajian_action_area] = 1  # 这里使用固定的位置进行lick，而不是像其他任务一样使用tracklength的某个比例
    return cueUsed, lick_label


def pickup_decorrelate(last_cueUsed, tracklength):
    """
    CS+（cue=0）对应far，CS-（cue=1）对应near。

    :param last_cueUsed: (bs)
    :param tracklength:
    :return:
        lick_label: (bs, x)
    """
    bs = last_cueUsed.shape[0]
    cueUsed = np.random.randint(0, 2, bs)
    lick_label = torch.zeros(bs, tracklength)
    for i in range(bs):
        if cueUsed[i] == 0:
            lick_label[i, int(0.9 * tracklength):tracklength] = 1
        if cueUsed[i] == 1:
            lick_label[i, int(0.7 * tracklength):int(0.8 * tracklength)] = 1
    return cueUsed, lick_label


def pickup_1234(last_cueUsed, tracklength):
    """
    对于每一个trail，根据线索类型决定每个位置上是否有奖励
    对于CS1234任务而言，只有CS1、CS2（cueUsed=0或2)的时候才需要在一个区间内lick

    :param last_cueUsed: (bs)
    :param tracklength:
    :return:
    """
    bs = last_cueUsed.shape[0]
    cueUsed = np.random.randint(0, 4, bs)
    lick_label = torch.zeros(bs, tracklength)
    for i in range(bs):
        if cueUsed[i] == 0 or cueUsed[i] == 1:
            lick_label[i, int(0.8 * tracklength):int(0.9 * tracklength)] = 1
    return cueUsed, lick_label


def pickup_lap(last_cueUsed, tracklength, cycle=4):
    """
    cueUsed实际上是此时跑的圈数 % cycle, lap任务和 Alter任务合并在一起了

    :param cycle: 周期。如果为4，就是传统的Lap任务；如果为2，就是传统的Alternative任务
    :param last_cueUsed: (bs)
    :param tracklength:
    :return:
    """
    bs = last_cueUsed.shape[0]
    cueUsed = (last_cueUsed + 1) % cycle  # 在之前的基础上增加一圈
    lick_label = torch.zeros(bs, tracklength)
    for i in range(bs):
        if cueUsed[i] == 0:
            lick_label[i, int(0.8 * tracklength):int(0.9 * tracklength)] = 1
    return cueUsed, lick_label


def oneCue_ec3_input_gen(cue, tracklength, cueLocation=0.05):
    """
    根据线索类型、（一个）线索位置，生成EC3的输入序列

    :param cueLocation: float，cueLocation在track上的位置比例
    :param cue: float Tensor(bs, ecnum)，相当于cuePattern[cueUsed, :]
    :param tracklength:
    :return: float Tensor(bs, tracklength, ecnum), 每个时刻、每个位置的每个ec3 Sensory输入
    """
    bs = cue.shape[0]
    ecnum = cue.shape[1]
    ec3input = torch.zeros(bs, tracklength, ecnum)

    # ec3input[:, int(cueLocation * tracklength), :] = cue  # 只在一个点进行cue刺激

    temp = list(range(10, 20))  # 在一个区间段内进行刺激，用于p01的描述模型
    ec3input[:, temp, :] = cue.unsqueeze(1).repeat(1, len(temp), 1)
    return ec3input


def evidence_cue_input_gen(bs, ecnum, cuePattern, tracklength, stimu_num=evidence_stimu_num, stimu_duration=evidence_stimu_duration):
    """
    (2*duration, 3*duration), (4*duration, 5*duration),,,,之间为刺激的位置。
    要注意，设置的刺激位置应该在决策位置之前。
    :param bs:
    :param ecnum:
    :param cuePattern: (2, ecnum)
    :param tracklength:
    :param stimu_num:总共给予多少个刺激
    :param stimu_duration: 每次刺激持续多少个时间步，参数去evidence_settings.py文件中读取
    :return:
        islick: (bs), int=0 or 1, 跟cueUsed作用差不多
        cue_left_right: (bs, stimu_num), int, 0, -1, 1
        ec3input: (bs, x, ecnum), tensor
        lick_label: (bs, x), tensor float
    """
    cue_left_right = np.zeros((bs, stimu_num))  # 每个位置是否在左侧有线索
    lick_label = torch.zeros(bs, tracklength)
    cueLocation = (1 + np.array(range(stimu_num))) * stimu_duration * 2  # 每次刺激的起点位置，(stimu_num)
    ec3input = np.zeros((bs, tracklength, ecnum))
    islick = [0] * bs
    for i in range(bs):
        for j in range(stimu_num):
            cue_left_right[i, j] = np.random.randint(-1, 1+1)  # -1, 0, 1
            # 如果是0的话，不给予EC3刺激
            if cue_left_right[i, j] == 1:
                ec3input[i, cueLocation[j] + range(stimu_duration), :] = np.tile(cuePattern[1
                                                                                 , :], (stimu_duration, 1))
            elif cue_left_right[i, j] == -1:
                ec3input[i, cueLocation[j] + range(stimu_duration), :] = np.tile(cuePattern[0
                                                                                 , :], (stimu_duration, 1))
        if sum(cue_left_right[i, :]) > 0:  # 1比-1更多则lick
            islick[i] = 1
            lick_label[i, int(0.8 * tracklength):int(0.9 * tracklength)] = 1

    return islick, cue_left_right, torch.Tensor(ec3input), lick_label


def evidence_poisson(bs, ecnum, cuePattern, tracklength, stimu_num=evidence_stimu_num, stimu_duration=evidence_stimu_duration):
    """
    以泊松的方式生成线索位置
    :param bs:
    :param ecnum:
    :param cuePattern: (2, ecnum)
    :param tracklength:
    :param stimu_num:总共给予多少个刺激
    :param stimu_duration: 每次刺激持续多少个时间步，参数去evidence_settings.py文件中读取
    :return:
        islick: (bs), int=0 or 1, 跟cueUsed作用差不多
        cue_left_right: (bs, stimu_num), int, 0, -1, 1
        ec3input: (bs, x, ecnum), tensor
        lick_label: (bs, x), tensor float
    """
    cue_left_right = np.zeros((bs, tracklength))  # 0，-1, 1
    lick_label = torch.zeros(bs, tracklength)
    ec3input = np.zeros((bs, tracklength, ecnum))
    islick = [0] * bs
    for i in range(bs):
        for x in range(int(0.1*tracklength), int(0.7*tracklength)):
            cue_left_right[i, x] = np.random.randint(0, 21)  # 0~14
            if cue_left_right[i, x] == 20:
                cue_left_right[i, x] = 1
                ec3input[i, x + np.array(range(stimu_duration)), :] += np.tile(cuePattern[1
                                                                                 , :], (stimu_duration, 1))
            elif cue_left_right[i, x] == 19:
                cue_left_right[i, x] = -1
                ec3input[i, x + np.array(range(stimu_duration)), :] += np.tile(cuePattern[0
                                                                                 , :], (stimu_duration, 1))
            else:  # 如果是0的话，不给予EC3刺激
                cue_left_right[i, x] = 0

        if sum(cue_left_right[i, :]) > 0:  # 1比-1更多则lick
            islick[i] = 1
            lick_label[i, int(0.8 * tracklength):int(0.9 * tracklength)] = 1

    return islick, cue_left_right, torch.Tensor(ec3input), lick_label


def sequence_gen(bs, ecnum, cuePattern, tracklength, stimunum=3, actnum=2):
    """

    :param bs:
    :param ecnum:
    :param cuePattern:
    :param tracklength:
    :param stimunum: 每个trail给予几次气味刺激，原文实验中是5
    :param actnum: 有几种行动结果（应该就是2）
    :return:
    """
    odornum = cuePattern.shape[0]  # 总共有多少种气味类型
    ordernum = 2  # 一次trail给予两次先后刺激
    cuelocation = (tracklength * torch.hstack(
        ((1 + torch.Tensor(range(stimunum))) * 0.1, torch.Tensor([0.6, 0.7])))).int()
    cue = torch.zeros(bs, stimunum + actnum, ecnum)  # 准备给予模型的线索刺激，包括记忆阶段的stimunum个和分辨阶段的actnum个
    cueUsed = torch.zeros(bs)
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
        cueUsed[sample] = (odorA_order > odorB_order) + 0.0

    num_ith_cue = 0
    bs = cue.shape[0]
    ecnum = cue.shape[2]
    ec3input = torch.zeros(bs, tracklength, ecnum)
    for x in range(tracklength):
        if num_ith_cue < stimunum + actnum and x == cuelocation[num_ith_cue]:
            ec3input[:, x:x + 8, :] = cue[:, num_ith_cue, :].unsqueeze(1)  # 每次刺激持续0.7秒
            num_ith_cue += 1

    lick_label = torch.zeros(bs, tracklength)
    for i in range(bs):
        if cueUsed[i] == 0:
            lick_label[i, int(0.8 * tracklength):int(0.9 * tracklength)] = 1
    return cueUsed, ec3input, lick_label


def envb_gen(bs, ecnum, cuePattern, tracklength):
    """
    每个轨道上的随机位置进行一次刺激，随后两秒后需要lick
    """
    cueUsed = np.random.randint(0, 10, bs)  # 这个值*5 = 在哪里给予刺激, 总共12种线索类型
    cueLocation = cueUsed * 5 + 10
    ec3input = np.zeros((bs, tracklength, ecnum))
    lick_label = torch.zeros(bs, tracklength)
    for sample in range(bs):
        ec3input[sample, cueLocation[sample]:cueLocation[sample]+10, :] = np.tile(cuePattern[0, :], (10, 1))
        lick_label[sample, cueLocation[sample]+20:cueLocation[sample]+30] = 1
    return cueUsed, torch.Tensor(ec3input), lick_label


def doublex(bs, ecnum, cuePattern, tracklength):
    """
    在 x0处给予刺激，要求 2*x0处进行lick
    """
    cueUsed = np.random.randint(0, 9, bs)
    cueLocation = (cueUsed+1) * 5
    ec3input = np.zeros((bs, tracklength, ecnum))
    lick_label = torch.zeros(bs, tracklength)
    for sample in range(bs):
        ec3input[sample, cueLocation[sample]:cueLocation[sample] + 10, :] = np.tile(cuePattern[0, :], (10, 1))
        lick_label[sample, 2*cueLocation[sample]:2*cueLocation[sample] + 10] = 1
    return cueUsed, torch.Tensor(ec3input), lick_label


def get_cueUsed_label_by_task(task, last_cueUsed, tracklength):
    """

    :param task:
    :param last_cueUsed: 上一次使用的cueUsed
    :param tracklength:
    :return:
    """
    if task == 'jiajian':
        cueUsed, lickLabel = pickup_jiajian(last_cueUsed, tracklength)  # CS+-的任务
    elif task == 'decorrelate':
        cueUsed, lickLabel = pickup_decorrelate(last_cueUsed, tracklength)  # decorrelate的任务
    elif task == 'lap':
        cueUsed, lickLabel = pickup_lap(last_cueUsed, tracklength, cycle=4)  # decorrelate的任务
    elif task == 'alter':
        cueUsed, lickLabel = pickup_lap(last_cueUsed, tracklength, cycle=2)  # decorrelate的任务
    elif task == '1234':
        cueUsed, lickLabel = pickup_1234(last_cueUsed, tracklength)
    else:
        cueUsed, lickLabel = None, None
        warnings.warn('Unknown task!')
    return cueUsed, lickLabel


def get_anything_by_task(task, bs, ecnum, cuePattern, last_cueUsed, tracklength):
    """
    直接获得每个线索、输入、舔的label。
    如果是Evidence任务，还有一个额外的cue_left_right
    bs: int
    ecnum: int
    last_cueUsed: (bs,), int
    tracklength: int
    :returns:
        focus_zone: 每个任务应该关注结果的轨道位置，list 80:90
        lick_label: (bs, x)
    """
    focus_zone = list(range(int(tracklength*0.8), int(tracklength*0.9)))  # 除了decorrelate任务以外，其他的任务基本都是这个
    if task == 'evidence':
        cueUsed, cue_left_right, ec3input, lick_label = evidence_cue_input_gen(bs, ecnum, cuePattern,
                                                                                    tracklength)  # cueUsed是01，是否应该舔
    elif task == 'evidence_poisson':
        cueUsed, cue_left_right, ec3input, lick_label = evidence_poisson(bs, ecnum, cuePattern, tracklength)
    elif task == 'sequence':
        cueUsed, ec3input, lick_label = sequence_gen(bs, ecnum, cuePattern, tracklength)
        cue_left_right = None
        focus_zone = list(range(int(tracklength * 0.8), int(tracklength * 0.9)))
    elif task == 'envb':
        cueUsed, ec3input, lick_label = envb_gen(bs, ecnum, cuePattern, tracklength)
        cue_left_right = None
        focus_zone = list(range(tracklength))  # 对于envb任务，哪里都有可能需要lick
    elif task == 'doublex':
        cueUsed, ec3input, lick_label = doublex(bs, ecnum, cuePattern, tracklength)
        cue_left_right = None
        focus_zone = list(range(tracklength))  # 对于envb任务，哪里都有可能需要lick
    else:
        cue_left_right = None  # 实际上就是什么都没有，只是在保存的时候用于占位
        cueUsed, lick_label = get_cueUsed_label_by_task(task, last_cueUsed, tracklength)
        if task == 'jiajian':
            focus_zone = list(jiajian_action_area)  # 因为加减任务的轨道长度是200，因此不能用默认的参数
        if task == 'lap' or task == 'alter':
            cue = torch.Tensor(np.tile(cuePattern[0, :], [bs, 1]))
        else:
            cue = cuePattern[cueUsed, :]  # 注意，这个操作之后似乎cue变量就变成了0和1的矩阵，需要用*.bool()还原回布尔变量
        ec3input = oneCue_ec3_input_gen(cue, tracklength)
        if task == 'lap' or task == 'alter':
            ec3input[cueUsed == 0, 90:100, :] = cuePattern[1, :]  # 在奖励之处给一个额外的刺激

        if task == 'decorrelate':
            focus_zone = list(range(int(tracklength * 0.7), int(tracklength * 0.8))) + list(range(int(tracklength * 0.9), int(tracklength * 1.0)))
    return cueUsed, ec3input, lick_label, cue_left_right, focus_zone


def eval_prepare(task, cueNum, bs, ec3_last, ec5_last, ca1_last, cueUsed, eval_bs=1000, loopnum=0):
    """
    在eval之前，对于数据的预处理。主要是将各类细胞的last数据处理好。
    eval_bs = 1000  # 用于eval的样本数量，越大越好
    loopnum: 如果是1，则是用于cs代码的eval设置；如果大于1，则是dv代码的eval设置
    """
    if task == 'lap' or task == 'alter':  # 单独准备Lap任务的初始值，因为每轮和每轮之间的cueUsed必须是连续的
        repeat_counts = 4 if task == 'lap' else 2
        first_index = [-1] * repeat_counts  # 搜索四个lap情况作为eval的细胞初值
        for i in range(repeat_counts):
            for index in range(bs):
                if cueUsed[index] == i:
                    first_index[i] = index
                    break
        assert all(x >= 0 for x in first_index)
        temp = int(eval_bs / repeat_counts)  # 复制每一行的次数
        if loopnum>=1:
            for loop in range(loopnum):  # 复制first_index的每一行并堆叠成(eval_bs, cellNum)的形式
                ec3_last[loop] = torch.vstack([ec3_last[loop][first_index[i], :].repeat(temp, 1) for i in range(repeat_counts)])
                ec5_last[loop] = torch.vstack([ec5_last[loop][first_index[i], :].repeat(temp, 1) for i in range(repeat_counts)])
                ca1_last[loop] = torch.vstack([ca1_last[loop][first_index[i], :].repeat(temp, 1) for i in range(repeat_counts)])
        else:
            ec3_last = torch.vstack([ec3_last[first_index[i], :].repeat(temp, 1) for i in range(repeat_counts)])
            ec5_last = torch.vstack([ec5_last[first_index[i], :].repeat(temp, 1) for i in range(repeat_counts)])
            ca1_last = torch.vstack([ca1_last[first_index[i], :].repeat(temp, 1) for i in range(repeat_counts)])
        last_cueUsed = np.repeat(np.array(range(repeat_counts)), repeats=temp)
    else:
        last_cueUsed = np.random.randint(0, cueNum, eval_bs)
        if loopnum>=1:
            for loop in range(loopnum):
                ec3_last[loop] = ec3_last[loop][0, :].repeat(eval_bs, 1)  # 以第一个元素的各个细胞的值，作为eval的初值
                ec5_last[loop] = ec5_last[loop][0, :].repeat(eval_bs, 1)
                ca1_last[loop] = ca1_last[loop][0, :].repeat(eval_bs, 1)
        else:
            ec3_last = ec3_last[0, :].repeat(eval_bs, 1)  # 以第一个元素的各个细胞的值，作为eval的初值
            ec5_last = ec5_last[0, :].repeat(eval_bs, 1)
            ca1_last = ca1_last[0, :].repeat(eval_bs, 1)

    return last_cueUsed, ec3_last, ec5_last, ca1_last


def get_cue_num(task):
    if task == 'lap' or task == '1234':
        cueNum = 4
    elif task == 'sequence':
        cueNum = 3
    elif task == 'envb':
        cueNum = 12  # 只是一小部分的线索，实际上有60个种类
    elif task == 'doublex':
        cueNum = 8  # 只是一小部分的线索，实际上有60个种类
    else:
        cueNum = 2
    return cueNum


def net_train(task, epochNum=500, bs=30, lr=0.003, given_net=None, given_cuePattern=None, isShuffleCA3=False,
              pre_losshis=None):
    """

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
    start_time = time.time()  # 单位；秒
    # 进行网络训练
    tracklength = 100
    ca1num = 100
    ecnum = 100
    actnum = 2
    if given_net is None:
        net = HPCrnn(ca1num=ca1num, ecnum=ecnum, tracklength=tracklength, actnum=actnum)
    else:
        net = given_net
        net.init_wca1act()
    if isShuffleCA3:
        net.shuffleCA3()

    # criterion = nn.CrossEntropyLoss()     # 普通的二分类
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 10]))  # 对于有奖励但没舔到时有非常大的惩罚，舔而没有奖励时则惩罚较小
    # 从结果上看，施加权重的loss训练效果更好，且lick行为符合生物的演变过程
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    '''初始化cue相关'''
    if task == 'lap' or task == '1234':
        cueNum = 4
    else:
        cueNum = 2  # jiajian, decorrelate, alter
    cueUsed = np.random.randint(0, cueNum, bs)  # 初始化一下，用于传递给之后的pickup函数
    if given_cuePattern is None:
        cuePattern = cue_gen_StimuNum(cueNum, ecnum)
    else:
        cuePattern = given_cuePattern
    cueUsedhis = np.zeros([epochNum, bs])
    ca1_all_his = torch.zeros(epochNum, tracklength, ca1num)
    ec5_all_his = torch.zeros(epochNum, tracklength, ecnum)
    ec3_all_his = torch.zeros(epochNum, tracklength, ecnum)
    ec3_last = torch.zeros(bs, ecnum)
    ec5_last = torch.rand(bs, ecnum)
    ca1_last = torch.zeros(bs, ca1num)
    losshis = torch.zeros(epochNum)
    actionhis = torch.zeros(epochNum, bs, tracklength)  # 用于绘制lick历史

    '''正式进行epoch迭代'''
    real_epochNum = epochNum
    loss = torch.Tensor([10000])
    for epoch in range(epochNum):
        # 如果loss已经很小了，就直接终止训练
        if loss.item() < 0.001:
            real_epochNum = epoch
            print('End Training.')
            break
        if task == 'evidence':
            cueUsed, cue_left_right, ec3input, lick_label = evidence_cue_input_gen(bs, ecnum, cuePattern, tracklength)
        elif task == 'evidence_poisson':
            cueUsed, cue_left_right, ec3input, lick_label = evidence_poisson(bs, ecnum, cuePattern, tracklength)
        else:
            cueUsed, lick_label = get_cueUsed_label_by_task(task, cueUsed, tracklength)
            if task == 'lap' or task == 'alter':
                cue = torch.Tensor(np.tile(cuePattern[0, :], [bs, 1]))
            else:
                cue = cuePattern[cueUsed, :]  # 注意，这个操作之后似乎cue变量就变成了0和1的矩阵，需要用*.bool()还原回布尔变量
            ec3input = oneCue_ec3_input_gen(cue, tracklength)
        cueUsedhis[epoch, :] = cueUsed
        optimizer.zero_grad()
        pred, ec3_this_his, ec5_this_his, ca1_this_his, ec3_last1, ec5_last1, ca1_last1 = net(ec3input, ec3_last,
                                                                                              ec5_last,
                                                                                              ca1_last)
        isLicked = torch.argmax(pred, 2) == 1  # (bs, tracklength)
        actionhis[epoch, :, :] = isLicked

        # 对actlist 和 licklabel进行展开以进行loss计算
        pred = pred.reshape(bs * tracklength, actnum)
        lick_label = lick_label.reshape(bs * tracklength)
        loss = criterion(pred, lick_label.long())
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)      # 梯度截断
        optimizer.step()

        ca1_all_his[epoch, :, :] = ca1_this_his  # epoch, x, i
        ec5_all_his[epoch, :, :] = ec5_this_his
        ec3_all_his[epoch, :, :] = ec3_this_his

        ec3_last = ec3_last1.detach()  # 虽然不知道为什么，但是这样就没有bug，用clone()的方法会有bug
        ec5_last = ec5_last1.detach()
        ca1_last = ca1_last1.detach()

        print('%d, %.5f' % (epoch, loss.item()))
        losshis[epoch] = loss
    duration_time = time.time() - start_time
    print('总训练时间：%.2f' % duration_time)
    if given_net is None:  # 只有在新训练模型的时候才保存模型
        torch.save({'task': task, 'cuePattern': cuePattern, 'net': net, 'losshis': losshis.data.numpy()}, 'hpc_rnn.pth')
    np.savez('cells.npz', ca1_all_his.data.numpy(), ec5_all_his.data.numpy(), ec3_all_his.data.numpy(), cueUsedhis,
             actionhis)
    plt.plot(losshis.data.numpy())
    if pre_losshis is not None:  # 加上以前的losshis的曲线（黑色），用于比较loss下降速度
        plt.plot(pre_losshis, color='black')
    plt.title('batch size: %d, lr: %.4f' % (bs, lr))
    plt.savefig('./fig_result/_loss result.jpg')
    plt.show()

    # 对模型进行测试，获得任务准确率，并将整个batch的结果保存
    net.eval()
    eval_bs = 1000
    last_cueUsed, ec3_last, ec5_last, ca1_last = \
        eval_prepare(task, cueNum, bs, ec3_last, ec5_last, ca1_last, cueUsed, eval_bs=eval_bs)
    cueUsed_eval, ec3input_eval, lick_label_eval, cue_left_right, focus_zone = \
        get_anything_by_task(task, eval_bs, ecnum, cuePattern, last_cueUsed, tracklength)  # 直接获取输入数据
    pred, ec3_last_his, ec5_last_his, ca1_last_his, _, _, _ = net(ec3input_eval, ec3_last, ec5_last, ca1_last,
                                                                  isSaveAll=True)
    pred = pred[:, focus_zone, :].reshape(eval_bs * len(focus_zone), actnum)  # 只关注focus_zone区域部分的准确率
    label_test = lick_label_eval[:, focus_zone].reshape(eval_bs * len(focus_zone))
    acc = torch.mean(((torch.argmax(pred, 1).reshape(eval_bs * len(focus_zone)) == label_test) + 0).float())
    print('accuracy: %.5f' % acc.float())
    np.savez('eval_result.npz', ca1_last_his.data.numpy(), ec5_last_his.data.numpy(), ec3_last_his.data.numpy(),
             cueUsed_eval, cue_left_right)
    print('evaluation done.')


if __name__ == '__main__':

    mode = 'training'  # 本次运行，是训练还是观察结果
    # mode = 'viewing'
    # mode = 'generalizing'  # 必须先train然后才能测试泛化

    # task = 'jiajian'  # 任务类型
    # task = 'decorrelate'
    # task = 'lap'
    # task = 'alter'
    # task = '1234'
    task = 'evidence'  # 目前没法学习3个以上的刺激，但DV代码可以

    epochNum = 500
    batch_size = 30
    lr = 0.003
    if task == 'lap' or task == '1234':
        cueNum = 4
    else:
        cueNum = 2

    seed = 6
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ——————————训练网络并保存网络结构、中间运行过程
    if mode == 'training':
        net_train(task, epochNum, batch_size, lr)

    # ——————————读取网络并测试和查看权重
    if mode == 'viewing':
        D = np.load('cells.npz')
        ca1his, ec5his, ec3his, cueUsedhis, actionhis = D['arr_0'], D['arr_1'], D['arr_2'], D['arr_3'], D['arr_4']
        train_his = ca1his  # 只选择一类细胞进行绘制

        # 获取eval的数据并进行accuracy分析, (bs, tracklength, ca1num)
        Deval = np.load('eval_result.npz')
        ca1_last_his, ec5_last_his, ec3_last_his, cueUsed_eval = Deval['arr_0'], Deval['arr_1'], Deval['arr_2'], Deval[
            'arr_3']
        eval_his = ca1_last_his

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
