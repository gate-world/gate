"""
一个示例程序，用来在训练中自动保存checkpoint，并在下次训练时自动读取、继续训练
used in : alternate_task.py
24.10.03
"""

import os
import numpy as np
import torch

if __name__=='__main__':
    cuenet = None
    optimizer = None

    # 先初始化各个对象
    epochnum = 100
    lossHis = np.zeros(epochnum)
    epochstart = 0

    # 加载用代码：
    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth')
        cuenet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochstart = checkpoint['epoch'] + 1
        lossHis = checkpoint['lossHis']

    # 训练的循环
    for epoch in range(epochstart, epochnum):
        print('train')

        # 保存模型断点：
        if epoch % 100 == 99:
            checkpoint = {
                'model': cuenet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'lossHis': lossHis
            }
            torch.save(checkpoint, 'checkpoint.pth')