"""
尝试使用HPCRnn实现条件刺激与奖励的关联
cue输入为CS+和CS-两类，agent需要在接收到CS+后在特定奖励位置产生lick行为
引自：Neural dynamics underlying associative  learning in the dorsal and ventral  hippocampus
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(5)


class HPCrnn(nn.Module):
    def __init__(self, ecnum, ca1num, ca3num, actnum, tracklength, ca3sigma=3):
        super(HPCrnn, self).__init__()
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.ca3num = ca3num
        self.actnum = actnum
        self.tracklength = tracklength
        self.sigma = ca3sigma
        self.CA3center = torch.linspace(0, self.tracklength, self.ca3num)

        self.Wec3ca1 = nn.Parameter(0.01 * torch.rand(ecnum, ca1num))
        self.Wca3ca1 = nn.Parameter(0.01 * torch.rand(ca3num, ca1num))
        self.Wca1ec5 = nn.Parameter(0.01 * torch.randn(ca1num, ecnum))
        self.Wec5ec3 = torch.eye(ecnum)
        self.Wca1act = nn.Parameter(0.01 * torch.rand(ca1num, actnum))

    def forward(self, bs, cue):
        ec3 = torch.zeros(bs, self.ecnum)
        ec5 = torch.rand(bs, self.ecnum)
        ca1 = torch.zeros(bs, self.ca1num)
        lick_train = torch.zeros(bs, self.tracklength)  # 将舔舐的动作记录下来

        for x in range(self.tracklength):
            if x == 10:
                sensoryInput = cue
            else:
                sensoryInput = 0
            ca3 = torch.exp(-((x - self.CA3center) / self.sigma) ** 2 / 2)
            ca1 = torch.relu(
                torch.matmul(ca3, self.Wca3ca1) * (1 + torch.sigmoid(torch.matmul(ec3, self.Wec3ca1))))
            ec5 = ec5 + torch.matmul(ca1, self.Wca1ec5)
            ec5 = 0.69 + 0.3 * torch.sigmoid(4 * (ec5 - 0.3))
            ec3 = ec5 * ec3 + sensoryInput
            action = torch.matmul(ca1, self.Wca1act)  # 相当于对所有CA1发放率加权得到一个是否lick的概率
            action = torch.sigmoid(action)
            lick_train[:, x] = action.squeeze()
        return lick_train


epochnum = 1000
bs = 1000
tracklength = 100
ecnum = 100
ca1num = 100
ca3num = 100
actnum = 1
cs_p = (torch.rand(1, ecnum) < 0.2).long().float()  # CS+
cs_m = (torch.rand(1, ecnum) < 0.2).long().float()  # CS-

net = HPCrnn(ecnum=ecnum, ca1num=ca1num, ca3num=ca3num, actnum=actnum, tracklength=tracklength)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
lossHis = torch.zeros(epochnum)

for epoch in range(epochnum):
    CS_list = (torch.rand(bs) > 0.5).long().float()  # 其中1表示CS+,0表示CS-
    cue = torch.zeros(bs, ecnum)
    lick_target = torch.zeros(bs, tracklength)
    for k in range(bs):
        cue[k, :] = cs_p if CS_list[k] else cs_m
        if CS_list[k]:
            lick_target[k, 59:79] = 1
    lick_train = net(bs=bs, cue=cue)
    optimizer.zero_grad()
    loss = criterion(lick_train, lick_target)
    loss.backward()
    optimizer.step()
    print(f'epoch:{epoch},loss of model is:{loss}')
    lossHis[epoch] = loss

plt.plot(lossHis.data.numpy())
plt.show()
print('end')
