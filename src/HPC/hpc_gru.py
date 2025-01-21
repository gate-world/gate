"""
用gru的方式来实现HPC，对比一下我的模型和GRU在cueAB任务下的表现
24.07.20

实验结论：如果sensory与action之间相距较远的话，没法训练
"""

import numpy as np
from matplotlib import pyplot as plt

from src.RNN.gru import *

if __name__ == '__main__':
    epochNum = 400
    tracklength = 100
    ca1num = 100
    ecnum = 100
    net = GRU_classify(feature_size=ecnum, hidden_size=100, class_size=2)
    bs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    cueA = torch.rand(1, ecnum) < 0.2  # 两种刺激ec3的方式都是随机的
    cueB = torch.rand(1, ecnum) < 0.2
    ca1his = torch.zeros(epochNum, tracklength, bs, ca1num)
    ec5his = torch.zeros(epochNum, tracklength, bs, ecnum)
    ec3his = torch.zeros(epochNum, tracklength, bs, ecnum)
    losshis = np.zeros(epochNum)
    for epoch in range(epochNum):
        isCueA = np.random.rand(bs) < 0.5
        cue = torch.zeros(bs, ecnum)
        for i in range(bs):
            if isCueA[i]:
                cue[i, :] = cueA
            else:
                cue[i, :] = cueB
        cue = torch.unsqueeze(cue, 1)  # (bs, 1, ecnum)
        cue = cue.expand(bs, 10, ecnum)  # 复制为这个大小的张量
        temp = torch.zeros(bs, int(0.8 * tracklength), ecnum)  # 填充sensory与action之间的空白。如果是70个step，则有概率能训练出来，80则完全没概率
        cue = torch.concat([cue, temp], 1)

        optimizer.zero_grad()
        # pred = net(cue)
        # 使用ec5.data.numpy()来观察
        pred = net(cue)

        loss = criterion(pred, torch.Tensor(isCueA).long())
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)      # 梯度截断
        optimizer.step()
        losshis[epoch] = loss.data.numpy()

        print(loss.item())

    # plt.plot(losshis)
    # plt.show()
    print('end')
