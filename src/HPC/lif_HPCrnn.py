'''
把HPCrnn迁移到LIF网络上，提供一个theta输入，观察相位进动现象
'''
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(5)


class LIF(nn.Module):
    def __init__(self, Vthresh, Vreset, beta=0.95):
        super(LIF, self).__init__()
        self.beta = beta
        self.Vthresh = Vthresh
        self.Vreset = Vreset
        self.spike_grad = self.approxgrad.apply

    def forward(self, V, I, S):
        '''
        :param V:membrane potential(t-1)
        :param I:input from upper layer(t)
        :param S:spike train(t-1)
        mathmodel: V(t) = beta*V(t-1) + Input(t) + Reset(spike or not)
        '''
        Vmem = self.beta * V + I + self.Vreset * S
        spk = self.spike_grad(V - self.Vthresh)
        return Vmem, spk

    class approxgrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Vmem):
            spk = (Vmem > 0).float()
            ctx.save_for_backward(Vmem)
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            Vmem, = ctx.saved_tensors
            grad = 1 / (1 + (np.pi * Vmem).pow_(2)) * grad_output
            return grad


class LIFRNN(nn.Module):
    def __init__(self, ecnum=100, ca1num=100, ca3num=100, actnum=2, length=100, dt=0.01, Vthresh=1, Vreset=-1):
        super(LIFRNN, self).__init__()
        self.ecnum = ecnum
        self.ca1num = ca1num
        self.ca3num = ca3num
        self.actnum = actnum
        self.length = length
        self.dt = dt
        self.sigma = 5
        self.CA3center = torch.linspace(0, self.length, self.ca3num)
        self.tau = 35
        self.Vthresh = Vthresh
        self.Vreset = Vreset

        self.Wec3ca1 = nn.Parameter(0.01 * torch.rand(ecnum, ca1num))
        self.Wca3ca1 = nn.Parameter(0.01 * torch.rand(ca3num, ca1num))
        self.Wca1ec5 = nn.Parameter(0.01 * torch.randn(ca1num, ecnum))
        self.Wec5ec3 = torch.eye(ecnum)
        self.Wca1action = nn.Parameter(0.01 * torch.rand(ca1num, actnum))

    def forward(self, cue):
        # initial membrane potential and spike train
        ec3 = self.Vreset * torch.ones(bs, self.ecnum)  # membrane potential
        ec5 = self.Vreset * torch.ones(bs, self.ecnum)
        ca1 = self.Vreset * torch.ones(bs, self.ca1num)
        ec3spk = torch.zeros(bs, self.ecnum)
        ec5spk = torch.zeros(bs, self.ecnum)
        ca1spk = torch.zeros(bs, self.ca1num)
        stepnum = int(self.length / self.dt)
        for step in range(1, stepnum + 1):
            t = step * self.dt
            x = t  # assume speed is 1
            if x == 0.1 * self.length:
                sensoryInput = cue
            else:
                sensoryInput = 0

            # generate ca3 spike train according to poisson encode: P(spike) = 1-e^(-I)
            ca3fr = torch.exp(-((x - self.CA3center) / self.sigma) ** 2 / 2)
            P_ca3spk = 1 - torch.exp(-ca3fr)  # poisson encode
            ca3spk = (torch.rand(self.ca3num) < P_ca3spk).long().float()

            ec3, ec3spk = LIF(Vthresh=self.Vthresh, Vreset=self.Vreset)(V=ec3, I=torch.matmul(ec5spk,
                                                                                              self.Wec5ec3) + sensoryInput,
                                                                        S=ec3spk)
            ca1, ca1spk = LIF(Vthresh=self.Vthresh, Vreset=self.Vreset)(V=ca1, I=(
                        torch.matmul(ec3spk, self.Wec3ca1) + torch.matmul(ca3spk, self.Wca3ca1)), S=ca1spk)
            ec5, ec5spk = LIF(Vthresh=self.Vthresh, Vreset=self.Vreset)(V=ec5, I=torch.matmul(ca1spk, self.Wca1ec5),
                                                                        S=ec5spk)

        actCell = torch.matmul(ca1spk, self.Wca1action)
        return actCell


epochnum = 200
ecnum = 100
spikenet = LIFRNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(spikenet.parameters(), lr=0.001)

lossHis = torch.zeros(epochnum)

cueL = (torch.rand(1, ecnum) < 0.2).long().float()  # 固定20个左右的ec3神经元作为线索输入细胞
cueR = (torch.rand(1, ecnum) < 0.2).long().float()
for epoch in range(epochnum):
    bs = 100
    cue = torch.zeros(bs, ecnum)
    isL = torch.rand(bs) > 0.5
    for k in range(bs):
        cue[k, :] = cueL if isL[k] else cueR

    pred = spikenet(cue=cue)
    optimizer.zero_grad()
    loss = criterion(pred, torch.tensor(isL).long())
    loss.backward()
    optimizer.step()
    print(f'current epoch is:{epoch}, loss is:{loss}')
    lossHis[epoch] = loss

plt.plot(lossHis.data.numpy())
plt.show()
print('end')
