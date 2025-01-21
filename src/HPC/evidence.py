"""
尝试用HPCrnn实现对线索的计数学习
cue输入为重复的L-R线索序列，agent需要选择出现线索多的一边
引自: Geometry of abstract learned knowledge in  the hippocampus(2020)
关键参数:在生成线索序列的时候，有奖励端的线索数:无奖励端的线索数=8:2(7.7:2.3 in reference)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from torch.optim.lr_scheduler import StepLR
import torchmetrics

torch.manual_seed(5)


class HPCrnn(nn.Module):
    def __init__(self, ecnum, ca1num, ca3num, actnum, tracklength, ca3sigma=5):
        super(HPCrnn, self).__init__()
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.ca3num = ca3num
        self.actnum = actnum
        self.tracklength = tracklength
        self.sigma = ca3sigma
        self.CA3center = torch.linspace(0, self.tracklength, self.ca3num)
        self.cueL = (torch.rand(1, ecnum) < 0.2).long().float()  # 固定20个左右的ec3神经元作为线索输入细胞
        self.cueR = (torch.rand(1, ecnum) < 0.2).long().float()

        self.Wec3ca1 = nn.Parameter(0.01 * torch.rand(ecnum, ca1num))
        self.Wca3ca1 = nn.Parameter(0.01 * torch.rand(ca3num, ca1num))
        self.Wca1ec5 = nn.Parameter(0.01 * torch.randn(ca1num, ecnum))
        self.Wec5ec3 = torch.eye(ecnum)
        self.Wca1act = nn.Parameter(0.01 * torch.rand(ca1num, actnum))
        self.ca1bias = nn.Parameter(0.01 * torch.rand(ca1num))

    def forward(self, bs, cue_train, ec3_last, ec5_last, ca1_last):
        ec3 = ec3_last  # 继承上一轮的信息
        ec5 = ec5_last
        ca1 = ca1_last
        ca1his = torch.zeros(self.tracklength, bs, self.ca1num)  # 记录本轮CA1表征：position * batch_size * ca1num
        for x in range(self.tracklength):
            sensoryInput = torch.zeros(bs, self.ecnum)
            for k in range(bs):
                if cue_train[k, x] == 1:
                    sensoryInput[k, :] = self.cueL  # 第k个trial位置x的线索指标为1
                elif cue_train[k, x] == -1:
                    sensoryInput[k, :] = self.cueR
                else:
                    sensoryInput[k, :] = 0
            ca3 = torch.exp(-((x - self.CA3center) / self.sigma) ** 2 / 2)
            ca1 = torch.relu(
                torch.matmul(ca3, self.Wca3ca1) * (1 + torch.sigmoid(torch.matmul(ec3, self.Wec3ca1))) - self.ca1bias)
            ec5 = ec5 + torch.matmul(ca1, self.Wca1ec5)
            ec5 = 0.69 + 0.3 * torch.sigmoid(4 * (ec5 - 0.3))
            ec3 = ec5 * ec3 + sensoryInput

            ca1his[x, :, :] = ca1

        actCell = torch.matmul(ca1, self.Wca1act)
        return actCell, ca1his, ec3, ec5, ca1


epochnum = 250
bs = 1000
tracklength = 100
labda1 = 0.08  # 原文中，奖励端每米平均有7.7个线索，非奖励端每米平均有2.3个线索，也就是lambda为0.08和0.02的泊松过程
labda2 = 0.02
ecnum = 100
ca1num = 100
ca3num = 100
actnum = 2
ec3_last = torch.zeros(bs, ecnum)
ec5_last = torch.rand(bs, ecnum)
ca1_last = torch.zeros(bs, ca1num)

net = HPCrnn(ecnum=ecnum, ca1num=ca1num, ca3num=ca3num, actnum=actnum, tracklength=tracklength)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
lossHis = torch.zeros(epochnum)
accuracyHis = torch.zeros(epochnum)

'record'
ca1_all_his = torch.zeros(epochnum, tracklength, bs, ca1num)
cue_his = torch.zeros(epochnum, bs)
cue_train_his = torch.zeros(epochnum, bs, tracklength)
wca3ca1_dict = {}

for epoch in range(epochnum):
    'isL决定每个trial的目标方向，生成一个线索列'
    isL = (torch.rand(bs) > 0.5).long().float()  # 其中1表示奖励端为左，否则奖励端为右
    cue_train = torch.zeros(bs, tracklength)  # 线索列有三种状态：1表示cueL，-1表示cueR，0表示没有线索
    poisson_labda1 = torch.poisson(torch.full((bs, tracklength), labda1))  # 生成奖励端线索的数量
    poisson_labda2 = torch.poisson(torch.full((bs, tracklength), labda2))  # 生成非奖励端线索的数量
    for i in range(bs):
        for j in range(tracklength):
            if poisson_labda1[i, j] > 0 and isL[i] == 1:
                cue_train[i, j] = 1  # 奖励端线索且奖励端为左
            elif poisson_labda2[i, j] > 0 and isL[i] == 0:
                cue_train[i, j] = 1  # 非奖励端线索且奖励端为右
            elif poisson_labda2[i, j] > 0 and isL[i] == 1:
                cue_train[i, j] = -1  # 非奖励端线索且奖励端为左
            elif poisson_labda1[i, j] > 0 and isL[i] == 0:
                cue_train[i, j] = -1  # 奖励端线索且奖励端为右
    cue_train_his[epoch, :, :] = cue_train
    pred, ca1his, ec3_last1, ec5_last1, ca1_last1 = net(bs=bs, cue_train=cue_train, ec3_last=ec3_last,
                                                        ec5_last=ec5_last, ca1_last=ca1_last)
    ca1_all_his[epoch, :, :, :] = ca1his
    optimizer.zero_grad()
    loss = criterion(pred, torch.tensor(isL).long())
    loss.backward()
    optimizer.step()
    net.Wca3ca1.data = torch.relu(net.Wca3ca1.data)
    scheduler.step()
    ec3_last = ec3_last1.detach()
    ec5_last = ec5_last1.detach()
    ca1_last = ca1_last1.detach()
    print(f'epoch:{epoch}, loss of model is:{loss}')
    lossHis[epoch] = loss
    cue_his[epoch, :] = isL
    wca3ca1_dict[f'epoch_{epoch}'] = net.Wca3ca1.data.cpu().numpy()

ca1_response = ca1_all_his[epochnum - 1, :, :, :].detach().numpy()  # 只保留最后一个epoch的ca1表征
cue_his = cue_his.detach().numpy()
cue_train_his = cue_train_his.detach().numpy()
savemat('ca1_response.mat', {'tensor': ca1_response})
savemat('trial_target.mat', {'tensor': cue_his})
savemat('trial_cue_train.mat', {'tensor': cue_train_his})
savemat('wca3ca1.mat', wca3ca1_dict)
plt.plot(lossHis.data.numpy())
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
print('end')
