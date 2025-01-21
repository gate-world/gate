"""
偶尔会遇到内存不够，程序崩溃的情况，但是引入断点保存之后好了很多
能够在400多轮的时候有一个loss的突然下降，目前还不知道为什么

"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

torch.manual_seed(5)


# class HPCrnn(nn.Module):
#     def __init__(self, ts=0.1, ecnum=100, ca3num=100, ca1num=100, actnum=2, ca3sigma=5):
#         super(HPCrnn, self).__init__()
#         self.ts = ts
#         self.ecnum = ecnum
#         self.ca3num = ca3num
#         self.actnum = actnum
#         self.ca1num = ca1num
#         self.ca3sigma = ca3sigma
#         self.tracklength = 100
#         self.ca1bias = nn.Parameter(torch.zeros(ca1num))
#         self.ec5bias = nn.Parameter(torch.zeros(ecnum))
#         self.actbias = nn.Parameter(torch.zeros(actnum))
#         self.wec3ca1 = nn.Parameter(torch.randn(ecnum, ca1num)*0.01)
#         self.wca3ca1 = nn.Parameter(torch.rand(ca3num, ca1num)*0.01)
#         self.wca1ec5 = nn.Parameter(torch.randn(ca1num, ecnum)*0.01)
#         self.wca1act = nn.Parameter(torch.randn(ca1num, actnum)*0.01)
#
#     def getCA3output(self, x):
#         ca3center = torch.linspace(0, self.tracklength, self.ca3num)
#         return torch.exp(-(ca3center-x) ** 2 / self.ca3sigma ** 2 / 2)
#
#     def forward(self, cue):
#         ec3 = torch.zeros(bs, self.ecnum)
#         ec5 = torch.rand(bs, self.ecnum)
#         ca1 = torch.zeros(bs, self.ca1num)
#
#         for x in range(self.tracklength):
#             if x <= 1:
#                 sensoryInput = cue
#             else:
#                 sensoryInput = 0
#
#             ca3 = self.getCA3output(x)
#             ca1 = torch.relu(torch.matmul(ca3, self.wca3ca1) * (1+torch.sigmoid(torch.matmul(ec3, self.wec3ca1)))-self.ca1bias)
#             ec5 = ec5 + 10*self.ts*torch.matmul(ca1, self.wca1ec5)+self.ec5bias
#             ec5 = 0.5 + 0.5*torch.sigmoid(4*(ec5-0.5))
#             ec3 = ec5*ec3 + sensoryInput
#
#         actCell = torch.matmul(ca1, self.wca1act)+self.actbias
#         return actCell


class HPCrnn(nn.Module):
    def __init__(self, ecnum=100, ca1num=100, ca3num=100, actnum=2, length=100):
        super(HPCrnn, self).__init__()
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.ca3num = ca3num
        self.actnum = actnum

        self.length = length
        self.sigma = 5
        self.CA3center = torch.linspace(0, self.length, self.ca3num)
        self.Wapical = nn.Parameter(0.01 * torch.rand(ecnum, ca1num))
        self.Wbasal = nn.Parameter(0.01 * torch.rand(ca3num, ca1num))
        self.Wca1ec5 = nn.Parameter(0.01 * torch.randn(ca1num, ecnum))
        self.Wec5ec3 = torch.eye(ecnum)
        self.Waction = nn.Parameter(0.01 * torch.rand(ca1num, actnum))
        self.ca1bias = nn.Parameter(torch.zeros(ca1num))
        self.ec5bias = nn.Parameter(torch.zeros(ecnum))
        self.actbias = nn.Parameter(torch.zeros(actnum))

    def forward(self, cue):
        ec5 = torch.rand(bs, self.ecnum)
        ec3 = torch.zeros(bs, self.ecnum)
        ca1 = torch.zeros(bs, self.ca1num)

        for x in range(1, 101):
            if x <= 1:
                sensoryInput = cue
            else:
                sensoryInput = 0
            ca3 = torch.exp(-((x - self.CA3center) / self.sigma) ** 2 / 2)
            # ca1 tuft部分必须限制大小，否则可能导致最终结果过大
            ca1 = torch.relu(
                torch.matmul(ca3, self.Wbasal) * (1 + torch.sigmoid(torch.matmul(ec3, self.Wapical))) - self.ca1bias)

            ec5 = ec5 + torch.matmul(ca1, self.Wca1ec5)
            ec5 = torch.clip(ec5, 0, 1)
            ec3 = torch.clip(ec5 * ec3 + sensoryInput, 0, 1)
            # ec3 = torch.clip(sensoryInput)
        actCell = torch.matmul(ca1, self.Waction)
        return actCell


class cuenet(nn.Module):
    def __init__(self, actnum=2, hidden_size=64, cue_size=100):
        super(cuenet, self).__init__()
        self.fc1 = nn.Linear(actnum, hidden_size)
        self.fc2 = nn.Linear(hidden_size, cue_size)

    def forward(self, actCell):
        x = torch.relu(self.fc1(actCell))
        sensoryInput = torch.sigmoid(self.fc2(x))
        return sensoryInput


epochnum = 800
stepnum = 100
ecnum = 100

alternatenet = HPCrnn()
cuenet = cuenet()
criterion1 = nn.CrossEntropyLoss()  # 交叉熵损失处理二分类问题，作为alternatenet根据线索推断的误差
criterion2 = nn.MSELoss()  # 均方损失处理线索生成问题，作为cuenet根据action生成线索的误差
optimizer1 = torch.optim.Adam(alternatenet.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(cuenet.parameters(), lr=0.001)

lossHis = torch.zeros(epochnum)  # 记录整个实验过程中的损失
epochstart = 0
if os.path.exists('checkpoint.pth'):
    # 加载用代码：
    checkpoint = torch.load('checkpoint.pth')
    cuenet.load_state_dict(checkpoint['model1'])
    alternatenet.load_state_dict(checkpoint['model2'])
    optimizer1.load_state_dict(checkpoint['optimizer1'])
    optimizer2.load_state_dict(checkpoint['optimizer2'])
    epochstart = checkpoint['epoch'] + 1
    lossHis = checkpoint['lossHis']

"一个自然的想法是通过两个网络实现alternate任务，一个网络通过EC-HPC循环网络将线索保持下来，并根据线索做出动作的决策；另一个网络通过一个mlp生成线索"
"每个epoch包含一个100次的交替转向任务，尽可能模拟真实动物实验中的操作：每个epoch起始阶段给定一个初始化条件，需要agent根据这个初始化条件自发地完成后续的任务"

cueL = (torch.rand(1, ecnum) < 0.2).long().float()  # 固定20个左右的ec3神经元作为线索输入细胞
cueR = (torch.rand(1, ecnum) < 0.2).long().float()
for epoch in range(epochstart, epochnum):
    'pretrain:先进行一段cue-action任务，训练alternate网络，使其能够根据cue做出正确的action'
    if epoch <= 0:
        bs = 100  # batch_size
        cue = torch.zeros(bs, ecnum)
        initact = torch.rand(bs) > 0.5  # 随机生成的初始化initial_action,0表示L，1表示R
        for k in range(bs):
            cue[k, :] = cueL if initact[k] else cueR  # L对应的线索为cueR，R对应的线索为cueL

        pred = alternatenet(cue)
        optimizer1.zero_grad()
        loss = criterion1(pred, torch.tensor(~initact).long())  # note：initact和真实类别刚好相反
        loss.backward()
        optimizer1.step()
        print(f'pretrain epoch:{epoch}, loss of alternatenet is:{loss}')
        lossHis[epoch] = loss
    else:
        '引入cuenet，现在需要根据上一个epoch的action自动生成线索'
        bs = 1
        initact = 1 if torch.rand(1) >= 0.5 else 0  # 在第一次探索过程中，agent的行动被人为固定，这个初始行动将作为后续行动的线索
        lasttrial = torch.tensor([[1 - initact, initact]], dtype=torch.long).float()  # lasttrial是一个二元张量，1表示历史action
        target = torch.arange(stepnum)  # tensor[0,...,99]
        target = ((target + initact + 1) % 2).long()  # 若initact=1，则初始target应该为0，target是未来100个step的目标行为

        predHis = torch.zeros(stepnum, 2)
        cueloss_sum = 0
        for step in range(stepnum):
            '每个step对cuenet进行训练'
            cuenet.train()
            alternatenet.eval()

            cue = cuenet(lasttrial)
            target_cue = torch.matmul(lasttrial, torch.cat([cueR, cueL], dim=0))  # 期望产生和cueL和cueR匹配的线索

            pred = alternatenet(target_cue)  # 根据线索生成action
            predHis[step, :] = pred
            max_index = torch.argmax(pred, dim=1)
            lasttrial = torch.zeros_like(pred, dtype=torch.long)
            lasttrial[0, max_index] = 1
            lasttrial = lasttrial.float()

            optimizer2.zero_grad()
            cueloss = criterion2(cue, target_cue)
            cueloss.backward()
            optimizer2.step()
            cueloss_sum += cueloss

        cuenet.eval()
        alternatenet.train()
        optimizer1.zero_grad()
        loss = criterion1(predHis, target)
        loss.backward()
        optimizer1.step()
        print(f'epoch is:{epoch},loss of cuenet is:{cueloss_sum}, loss of alternatenet is:{loss}')
        lossHis[epoch] = loss
        if epoch % 100 == 99:
            # 保存模型断点：
            checkpoint = {
                'model1': cuenet.state_dict(),
                'model2': alternatenet.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'epoch': epoch,
                'lossHis': lossHis
            }
            torch.save(checkpoint, 'checkpoint.pth')

            # sns.heatmap(predHis.detach().numpy(), cmap='coolwarm', cbar=True)
            # plt.ylabel('step')
            # plt.title(f'prediction of epoch{epoch},initial action is{initact}')
            # plt.show()

plt.plot(lossHis.data.numpy())
plt.show()
