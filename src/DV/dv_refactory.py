"""
用不应期的方式来让ec3具有动量/速度性质，被判定进行shift的细胞，在短时间内无法再次shift
或许这会能够学习到Lap、Alter任务了吧
from: dv_gating.py
24.10.02

目前来看，似乎这个模型还是有潜力完成Alternative和Lap任务的。
因为“完成了”对于OnOff的群体刻画，只是需要在不同的保持阶段中，使用不同的输入
（在On阶段使用 EC3input = 0， 在 Off阶段使用 EC3input = 0.5），希望网络能够学到这一点
但是使用的 P01和 P10的参数与原来不一样了，现在具有更大的 tau_nax, 更小的 tau_min
参考 241003
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.lick import cs
from src.tools import myplot


class HPCSingleLoop(nn.Module):
    def __init__(self, bs, ts=0.1, ecnum=100, ca1num=100, ca3num=100, ca3sigma=5, tracklength=100, is_continue=True):
        """
        单层的Loop。注意每次的forward是在一个x处运行一次迭代，而不是走完整个tracklength

        :param ts: 时间步长，单位为秒
        :param ecnum: EC3 & EC5的细胞数量相同，一一对应
        :param ca1num:
        :param ca3sigma: 注意需要比较大（>=5），否则训练可能非常缓慢
        :param is_continue: 是否用默认的连续方式进行运算。否则的话，转用Markov的离散模式，模仿EC3进行运算
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
        self.lambda_ec5 = 0.1  # 对EC5施加的正则化的强度
        self.wca3ca1 = nn.Parameter(torch.rand(ca3num, ca1num) * 0.01)
        self.wec3ca1 = nn.Parameter(torch.randn(ecnum, ca1num) * 0.01)
        self.wca1ec5 = nn.Parameter(torch.randn(ca1num, ecnum) * 0.01)
        self.wec5ec3 = nn.Parameter(torch.randn(ecnum, ecnum) * 0.01)  # 没什么卵用，并不会加速训练，直接扔了

        nn.init.xavier_normal_(self.wca1ec5)
        nn.init.xavier_normal_(self.wec3ca1)

        self.is_continue = is_continue
        self.multiple_factor = 1

        self.refactory_duration = 20  # 其实这里有点太高了，，，但是MATLAB的仿真结果表明，如果不应期的持续时间=1效果也不好
        self.refactory_pointer = 0
        self.shift_counter_01 = torch.zeros(bs, ecnum, self.refactory_duration)
        self.shift_counter_10 = torch.zeros(bs, ecnum, self.refactory_duration)

    def shift_to_discrete(self, multiple_factor=1):
        """
        将原本的连续性模型转变为离散Markov模型，每个细胞用若干个离散细胞来替代
        :param multiple_factor: 每一个原连续的EC3指数衰减，需要使用多少个离散的On Off细胞进行仿真
        :return:
        """
        self.is_continue = False
        self.eval()
        self.multiple_factor = multiple_factor

    def detach_refactory(self):
        torch.detach_(self.shift_counter_10)
        torch.detach_(self.shift_counter_01)

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

    def sigp01(self, t):
        """
        P10在P01之上。
        EC3 On的概率，收敛值为 sigp01 / (sigp01 + sigp10), tau值为 1./(sigp01+sigp10)

        t为input，即Sensory输入 + 上游dorsal CA1输入 + EC5输入;
        输入t较小时，拒绝input写入并保持信息（tau大，stationary小）;
        输入t中等时，拒绝input写入并遗忘（tau小， stationary大）;
        输入t较大时，允许input写入并遗忘 (tau小，stationary大）.
        """
        return 0.00001 + 2.8 * torch.sigmoid(4 * (t - 1.5))  # 0.03代表惊喜转换率，0.8代表最大的转换速率，可以大于1

    def sigp10(self, t):
        return 0.001 + 2.6 * torch.sigmoid(10 * (t - 0.5))  # 0.01代表静息最小遗忘率，0.6代表最大的转换速率，可以大于1

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
            ca1 = torch.relu(
                torch.sigmoid(10 * (torch.matmul(ca3, self.wca3ca1) - 0.5))
                * (1 + 3 * torch.sigmoid(torch.matmul(ec3, self.wec3ca1)))
                - self.ca1bias)

            # 注意这里取消了EC5的Bias，如果需要就只能依靠CA1产生，对训练结果没有负面影响
            ec5 = ec5 + self.ts * torch.matmul(ca1, self.wca1ec5) + self.ec5bias
            ec5 = ec5 - self.lambda_ec5 * (ec5 - 0.5)  # 对EC5施加正则化，让EC5趋于0.5
            ec5 = torch.clip(ec5, 0, 1)  # 可以考虑给EC5加上一个正则化，让系统有一个“遗忘倾向”；或者用不断变化的bias来模仿搜索关联学习过程

            ec3_all_input = ec3input + ec5  # Sensory+ec5，或者 dCA1+ec5
            rate01 = self.sigp01(ec3_all_input) * self.ts  # 等价于dr/dt = (1-r)*sigp01 - r*sigp10
            rate10 = self.sigp10(ec3_all_input) * self.ts

            read_refactor_pointers = [i for i in range(self.refactory_duration) if i != self.refactory_pointer]  # 需要排除哪些index的细胞计算shift
            ec3_shifting_01 = rate01 * (1 - ec3 - torch.sum(self.shift_counter_10[:, :, read_refactor_pointers], dim=2))  # 本次转变的细胞量
            ec3_shifting_10 = rate10 * (ec3 - torch.sum(self.shift_counter_01[:, :, read_refactor_pointers], dim=2))
            self.shift_counter_01[:, :, self.refactory_pointer] = ec3_shifting_01  # 本轮中有多少细胞进行了shift
            self.shift_counter_10[:, :, self.refactory_pointer] = ec3_shifting_10
            ec3 = ec3 + ec3_shifting_01 - ec3_shifting_10 + \
                  0.02 * 0.3 * torch.randn_like(ec3)  # 加上了一个噪声，强度和仿真markov的一致
            self.refactory_pointer = (self.refactory_pointer + 1) % self.refactory_duration

        else:  # EC3离散模式, 此时的EC3为：(bs, ecnum, multiple_factor)
            ca1 = torch.relu(
                torch.sigmoid(10 * (torch.matmul(ca3, self.wca3ca1) - 0.5))
                * (1 + 3 * torch.sigmoid(torch.matmul(torch.mean(ec3.float(), dim=2), self.wec3ca1)))
                - self.ca1bias)

            # 注意这里取消了EC5的Bias，如果需要就只能依靠CA1产生，对训练结果没有负面影响
            ec5 = ec5 + self.ts * torch.matmul(ca1, self.wca1ec5) + self.ec5bias
            ec5 = ec5 - self.lambda_ec5 * (ec5 - 0.5)  # 对EC5施加正则化，让EC5趋于0.5
            ec5 = torch.clip(ec5, 0, 1)  # 可以考虑给EC5加上一个正则化，让系统有一个“遗忘倾向”；或者用不断变化的bias来模仿搜索关联学习过程

            ec3_all_input = ec3input + ec5  # Sensory+ec5，或者 dCA1+ec5
            rate01 = self.sigp01(ec3_all_input) * self.ts  # 等价于dr/dt = (1-r)*sigp01 - r*sigp10
            rate10 = self.sigp10(ec3_all_input) * self.ts
            # 大小为：（bs，ecnum, multiple_factor)
            trans01 = torch.rand(bs, self.ecnum, self.multiple_factor) < rate01.unsqueeze(2).expand(bs, self.ecnum, self.multiple_factor)
            trans10 = torch.rand(bs, self.ecnum, self.multiple_factor) < rate10.unsqueeze(2).expand(bs, self.ecnum, self.multiple_factor)
            new_state = ec3
            new_state[torch.logical_and(torch.logical_not(ec3), trans01)] = True
            new_state[torch.logical_and(ec3, trans10)] = False

        return ec3, ec5, ca1


class HPC_DV(nn.Module):
    def __init__(self, bs, ts=0.1, ecnum=100, ca1num=100, ca3sigma=5, tracklength=100, loopnum=2, actnum=2):
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
            self.loopList.append(HPCSingleLoop(bs, ts=ts, ca1num=ca1num, ecnum=ecnum, tracklength=tracklength))
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

    def shift_to_discrete(self, multiple_factor=1):
        self.eval()
        for i in range(self.loopnum):
            self.loopList[i].shift_to_discrete(multiple_factor=multiple_factor)

    def detach_refactory(self):
        """
        必须在每次detach的时候调用这个，用来防止shift_counter变量被重复计算梯度
        :return:
        """
        for i in range(self.loopnum):
            self.loopList[i].detach_refactory()

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
    tracklength = 200 if task == 'jiajian' else 100
    ca1num = 100
    ecnum = 100
    actnum = 2

    '''初始化网络'''
    if given_net is None:
        net = HPC_DV(bs, loopnum=loopnum, ca1num=ca1num, ecnum=ecnum, tracklength=tracklength, actnum=actnum)
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

    if task == 'lap':
        bp_counter = 5  # 如果设置为4的话，很容易就进入震荡了，设置为5稍微好点
    elif task == 'alter':
        bp_counter = 3  # 用于统计多少个trail之后进行一次BP，用这种方式来训练需要多个Trail的任务
    else:
        bp_counter = 1

    '''正式进行epoch迭代'''
    real_epochNum = epochNum
    loss = torch.Tensor([10000])
    for epoch in range(epochNum):
        # 如果loss已经很小了，就直接终止训练
        if loss.item() < 0.015:
            real_epochNum = epoch
            print('End Training.')
            break
        if task == 'evidence':
            cueUsed, is_cue_left, ec3input, lick_label = cs.evidence_cue_input_gen(bs, ecnum, cuePattern, tracklength)
        elif task == 'evidence_poisson':
            cueUsed, cue_left_right, ec3input, lick_label = cs.evidence_poisson(bs, ecnum, cuePattern,
                                                                                  tracklength)
        else:
            # 根据任务类型来确定reward位置
            cueUsed, lick_label = cs.get_cueUsed_label_by_task(task, cueUsed, tracklength)
            if task == 'lap' or task == 'alter':
                cue = torch.Tensor(np.tile(cuePattern[0, :], [bs, 1]))
            else:
                cue = cuePattern[cueUsed, :]  # 注意，这个操作之后似乎cue变量就变成了0和1的矩阵，需要用*.bool()还原回布尔变量
            ec3input = cs.oneCue_ec3_input_gen(cue, tracklength)
        cueUsedhis[epoch, :] = cueUsed
        optimizer.zero_grad()
        pred, ec3_this_his, ec5_this_his, ca1_this_his, ec3_last, ec5_last, ca1_last \
            = net(ec3input, ec3_last, ec5_last, ca1_last, isSaveAllBatch=False)
        isLicked = torch.argmax(pred, 2) == 1  # (bs, tracklength)
        actionhis[epoch, :, :] = isLicked

        pred = pred.reshape(bs * tracklength, actnum)
        lick_label = lick_label.reshape(bs * tracklength)
        loss = criterion(pred, torch.Tensor(lick_label).long())
        if epoch % bp_counter == 0:  # 如果是Lap任务或者Alternative任务，则只在特定的epoch进行BP
            loss.backward()
            optimizer.step()
            net.detach_refactory()

        for loop in range(loopnum):
            ca1_all_his[epoch, loop, :, :] = ca1_this_his[loop, :, :]  # epoch, loop, x, cellnum, 只记录batch中第一个sample
            ec5_all_his[epoch, loop, :, :] = ec5_this_his[loop, :, :]
            ec3_all_his[epoch, loop, :, :] = ec3_this_his[loop, :, :]
            if epoch % bp_counter == 0:  # 如果是Lap任务或者Alternative任务，则只在特定的epoch进行梯度截断
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

    duration_time = time.time() - start_time
    print('总训练时间：%.2f' % duration_time)
    np.savez('cells.npz', ca1_all_his.data.numpy(), ec5_all_his.data.numpy(),
             ec3_all_his.data.numpy(), cueUsedhis, actionhis)  # (epochnum, loopnum, x, cell); cueUsedhis:(epochnum, bs)
    torch.save({'task': task, 'cuePattern': cuePattern, 'net': net, 'losshis': losshis.data.numpy()}, 'hpc_rnn.pth')
    plt.plot(losshis.data.numpy())
    if pre_losshis is not None:  # 加上以前的losshis的曲线（黑色），用于比较loss下降速度
        plt.plot(pre_losshis, color='black')
    plt.title('batch size: %d, lr: %.4f' % (bs, lr))
    plt.savefig('./fig_result/_loss result.jpg')
    plt.show()

    '''对模型进行测试，获得任务准确率，并将整个batch的结果保存'''
    net.eval()
    eval_bs = 1000  # 用于eval的样本数量，越大越好
    last_cueUsed, ec3_last, ec5_last, ca1_last = \
        cs.eval_prepare(task, cueNum, bs, ec3_last, ec5_last, ca1_last, cueUsed, eval_bs=eval_bs, loopnum=loopnum)
    cueUsed_eval, ec3input_eval, lick_label_eval, cue_left_right = \
        cs.get_anything_by_task(task, eval_bs, ecnum, cuePattern, last_cueUsed, tracklength)  # 直接获取输入数据
    pred, ec3_last_his, ec5_last_his, ca1_last_his, _, _, _ = net(ec3input_eval, ec3_last, ec5_last, ca1_last,
                                                                  isSaveAllBatch=True)
    pred = pred.reshape(eval_bs * tracklength, actnum)
    label_test = lick_label_eval.reshape(eval_bs * tracklength)
    acc = torch.mean(((torch.argmax(pred, 1).reshape(eval_bs * tracklength) == label_test) + 0).float())
    print('accuracy: %.5f' % acc.float())
    np.savez('eval_result.npz', ca1_last_his.data.numpy(), ec5_last_his.data.numpy(), ec3_last_his.data.numpy(),
             cueUsed_eval, cue_left_right)  # (eval_bs, loopnum, tracklength, cell); cueUsed:(eval_bs)
    print('evaluation done.')


if __name__ == '__main__':
    mode = 'training'  # 本次运行，是训练还是观察结果
    # mode = 'viewing'
    # mode = 'generalizing'  # 必须先train然后才能测试泛化
    # mode = 'discrete'  # 用离散的方式来描述EC3，用训练之后的网络来仿真；效果居然出奇的好

    # task = 'jiajian'  # 任务类型, 注意jiajian任务的轨道长度为200而非100，会导致默认的准确率就很高
    # task = 'decorrelate'
    # task = 'lap'
    task = 'alter'
    # task = '1234'
    # task = 'evidence'
    # task = 'evidence_poisson'

    loopnum = 3
    epochNum = 1000
    batch_size = 30
    lr = 0.03
    if task == 'lap' or task == '1234':
        cueNum = 4
    else:
        cueNum = 2

    seed = 4
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ——————————训练网络并保存网络结构、中间运行过程
    if mode == 'training':
        net_train(loopnum, task, epochNum, batch_size, lr)

    # ——————————读取网络并测试和查看权重
    if mode == 'viewing':
        loop_to_plot = 0  # 选择是第几个loop进行绘制
        cell_type_plot = 'ec3'  # 选择的是哪种细胞进行绘制

        D = np.load('cells.npz')
        # ca1his: (epoch, loop, x, cell); cueUsedhis: (epoch, bs); actionhis: (epoch, bs)
        ca1his, ec5his, ec3his, cueUsedhis, actionhis = D['arr_0'], D['arr_1'], D['arr_2'], D['arr_3'], D['arr_4']
        Deval = np.load('eval_result.npz', allow_pickle=True)  # 因为这里的cue_left_right有可能是None，因此需要允许pickle
        ca1_last_his, ec5_last_his, ec3_last_his, cueUsed_eval, cue_left_right = Deval['arr_0'], Deval['arr_1'], Deval['arr_2'], Deval[
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

        myplot.viewing(task, cueNum, train_his, actionhis, cueUsedhis, eval_his, cueUsed_eval, cue_left_right,is_plot_each_cell=True)

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
        net_train(loopnum, task, epochNum, batch_size, lr, given_net=net, pre_losshis=losshis)

    if mode == 'discrete':
        multiple_factor = 10  # 每个连续细胞用几个离散OnOff细胞替代。在一个离散细胞的情况下就已经准确率非常高了, 十个就基本上达到了最大值
        temp = torch.load('hpc_rnn.pth')
        net = temp['net']
        cuePattern = temp['cuePattern']
        ecnum = net.ecnum
        ca1num = net.ca1num
        actnum = net.actnum
        tracklength = net.tracklength
        eval_bs = 1000  # 用于eval的样本数量，越大越好
        net.shift_to_discrete(multiple_factor=multiple_factor)

        last_cueUsed = np.random.randint(0, cueNum, eval_bs)
        ec3_last = [torch.Tensor(1)] * loopnum  # loopnum, bs, ecnum————注意，这里使用的细胞初值都是0，如果用理想值的话准确率还能更高
        ec5_last = [torch.Tensor(1)] * loopnum
        ca1_last = [torch.Tensor(1)] * loopnum
        cueUsed_eval, ec3input_eval, lick_label_eval, cue_left_right, focus_zone = \
            cs.get_anything_by_task(task, eval_bs, ecnum, cuePattern, last_cueUsed, tracklength)  # 直接获取输入数据
        for loop in range(loopnum):
            ec3_last[loop] = torch.zeros(eval_bs, ecnum, multiple_factor).bool()
            ec5_last[loop] = torch.zeros(eval_bs, ecnum)
            ca1_last[loop] = torch.zeros(eval_bs, ca1num)

        # 此时的his变量是None，因为his的处理实在太麻烦了，细胞数量与连续模型的并不一致
        pred, ec3_last_his, ec5_last_his, ca1_last_his, _, _, _ = net(ec3input_eval, ec3_last, ec5_last, ca1_last,
                                                                      isSave=False)
        pred = pred[:, focus_zone, :].reshape(eval_bs * len(focus_zone), actnum)  # 只关注focus_zone区域部分的准确率
        label_test = lick_label_eval[:, focus_zone].reshape(eval_bs * len(focus_zone))
        acc = torch.mean(((torch.argmax(pred, 1).reshape(eval_bs * len(focus_zone)) == label_test) + 0).float())
        print('accuracy: %.5f' % acc.float())
        # np.savez('eval_result.npz', ca1_last_his.data.numpy(), ec5_last_his.data.numpy(), ec3_last_his.data.numpy(),
        #          cueUsed)  # (eval_bs, loopnum, tracklength, cell); cueUsed:(eval_bs)
        print('evaluation done.')

    print('end')

