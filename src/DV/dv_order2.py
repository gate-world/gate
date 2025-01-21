"""
尝试用新的DV模型来训练Order任务
24.10.06
from：dv_order.py
"""
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.tools import myplot
from src.DV.dv_gating import HPC_DV


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
            ec3input[:, x:x+7, :] = cue[:, num_ith_cue, :].unsqueeze(1)
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


def net_train(epochNum=500, bs=30, lr=0.003, lambda_ec5=0.002):
    # task = 'trace'  # 需要进行的任务类型，
    task = 'order'

    start_time = time.time()  # 单位；秒
    # 进行网络训练
    tracklength = 100
    ca1num = 100
    ecnum = 100
    actnum = 2
    loopnum = 1

    odornum = 2
    ordernum = 2  # 每个trail给予几次（顺序的）线索刺激
    # cuelocation = 0.12 * tracklength * (torch.Tensor(range(ordernum + actnum)) + 2)  # 多个刺激位置
    cuelocation = (tracklength * torch.hstack(((1+torch.Tensor(range(ordernum))) * 0.15, 0.5+torch.Tensor(range(actnum)) * 0.15))).int()
    net = HPC_DV(loopnum=loopnum, ca1num=ca1num, ecnum=ecnum, tracklength=tracklength, actnum=actnum)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

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
    labelhis = torch.zeros(epochNum)

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
        pred, ec3_this_his, ec5_this_his, ca1_this_his, ec3_last, ec5_last, ca1_last \
            = net(ec3input, ec3_last, ec5_last, ca1_last, isSaveAllBatch=False)
        pred = pred[:, -1, :]  # 只取轨道末端的结果，（bs, actnum)
        loss_ec5 = torch.zeros(bs, ecnum)  # 对EC5的二范数正则，会导致Dorsal CA1数量稀少，但EC5全都趋于0.5
        for loop in range(loopnum):
            loss_ec5 += (ec5_last[loop] - 0.5) ** 2
        loss_ec5 = torch.mean(loss_ec5)
        loss_lick = criterion(pred, torch.Tensor(label).long())
        loss = loss_lick + lambda_ec5 * loss_ec5
        loss.backward()
        optimizer.step()

        for loop in range(loopnum):
            ca1_all_his[epoch, loop, :, :] = ca1_this_his[loop, :, :]  # epoch, loop, x, ecnum
            ec5_all_his[epoch, loop, :, :] = ec5_this_his[loop, :, :]
            ec3_all_his[epoch, loop, :, :] = ec3_this_his[loop, :, :]
            ec3_last[loop] = ec3_last[loop].detach()
            ec5_last[loop] = ec5_last[loop].detach()
            ca1_last[loop] = ca1_last[loop].detach()

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
    lr = 0.01
    cueNum = 2

    # mode = 'training'  # 本次运行，是训练还是观察结果
    mode = 'viewing'

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

        history = ec3his[:, 0, :, :]  # 只选择一类细胞进行绘制，且只选择一个层的loop

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
