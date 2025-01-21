"""
transformer进行CS+-任务的学习。
非常快就学完了，倒是也在意料之中，，，
关于Lap任务，目前还没想好具体该怎么实现。毕竟需要更长时间的信息保持。

24.11.15
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from src.tools.myplot import my_save_fig
from src.tools.delete_file import delete_files_in_directory


# 定义简单的 Transformer 模型
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, model_dim, num_heads, num_layers, output_dim):
        """
        input_dim: 输入数据的维度，实际上是有多少种类的单词
        model_dim: 嵌入向量的维度
        """
        super(SimpleTransformer, self).__init__()
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embedding = nn.Embedding(input_dim, model_dim)  # 输入嵌入层
        self.positional_encoding = torch.randn(1, seq_len, model_dim)  # 假设最大序列长度为 10，这里也可以将其设置为parameter

        # Transformer 层
        self.transformer = nn.Transformer(d_model=model_dim,
                                          nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)

        self.fc_out = nn.Linear(model_dim, output_dim)  # 输出层

    def shuffle_ca3(self):
        """重新打乱位置编码"""
        self.positional_encoding = torch.randn(1, self.seq_len, self.model_dim)  # 假设最大序列长度为 10

    def shuffle_embed(self):
        self.embedding = nn.Embedding(self.input_dim, self.model_dim)  # 输入嵌入层

    def forward(self, src, tgt):
        # 将输入序列和目标序列通过嵌入层加上位置编码
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :].to(torch.device("cuda:0"))
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :].to(torch.device("cuda:0"))

        src = src.transpose(0, 1)  # Transformer 要求的输入格式: (seq_len, batch_size, model_dim)
        tgt = tgt.transpose(0, 1)  # Transformer 要求的输入格式: (seq_len, batch_size, model_dim)

        # 使用 Transformer 模型
        output = self.transformer(src, tgt)

        # 将输出维度变为 (batch_size, seq_len-1, output_dim)
        output = self.fc_out(output)  # 输出维度：(seq_len, batch_size, output_dim)
        output = output.transpose(0, 1)  # 转置为 (batch_size, seq_len, output_dim)

        return output


# 数据生成：输入是一个整数序列
def generate_jiajian(batch_size, seq_len, cueUsed_start):
    """
    cueUsed: 即n, 第一个元素决定了从哪里开始使用sensory
    """
    src = torch.zeros((batch_size, seq_len), dtype=torch.int)  # 输入是 0 到 input_dim-1 之间的整数，因为之后tgt要加上1
    label = (torch.rand(batch_size)<0.5).bool()
    src[label, 10:20] = 1
    src[:, 10:20] += cueUsed_start
    tgt = torch.zeros((batch_size, seq_len), dtype=torch.int)
    tgt[label, 90:100] = 1

    return src, tgt


# 生成一个Evidence的输入
def generate_evidence(batch_size, seq_len, cueUsed_start):
    src = torch.zeros((batch_size, seq_len), dtype=torch.int)  # 输入是 0 到 input_dim-1 之间的整数，因为之后tgt要加上1
    cue_r = torch.rand(batch_size, seq_len) < 0.1
    cue_l = torch.rand(batch_size, seq_len) < 0.1
    src[cue_r] = cueUsed_start
    src[cue_l] = cueUsed_start + 1
    src[:, 0:10] = 0
    src[:, 80:seq_len] = 0

    label = (torch.sum(cue_r.int()-cue_l.int(), dim=1) > 0).bool()
    tgt = torch.zeros((batch_size, seq_len), dtype=torch.int)
    tgt[label, 90:100] = 1

    return src, tgt


def training(task, input_dim, output_dim, batch_size=32, seq_len=100, num_epochs=40,
             model=None, shuffle_embed=False, shuffle_ca3=False, cueUsed=None):
    print(f'task: {task}')
    device = torch.device("cuda:0")
    # 创建模型
    if model is None:
        model = SimpleTransformer(input_dim, seq_len, model_dim=64, num_heads=2, num_layers=2, output_dim=output_dim)
    model.to(device)
    if shuffle_embed:
        model.shuffle_embed()
    if shuffle_ca3:
        model.shuffle_ca3()
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 对于整数预测，使用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练过程
    losshis = np.zeros(num_epochs)
    stop_epoch = num_epochs
    model.train()
    loss_stop = 100000
    for epoch in range(num_epochs):
        if loss_stop < 0.01 and stop_epoch == num_epochs:
            stop_epoch = epoch

        # 生成一个批次的数据
        cueUsed = cueUsed if cueUsed is not None else 0
        if task == 'jiajian':
            src, tgt = generate_jiajian(batch_size, seq_len, cueUsed)
        elif task == 'evidence':
            src, tgt = generate_evidence(batch_size, seq_len, cueUsed)

        # 在目标序列中，将第一个元素作为解码器输入，其余作为标签
        tgt_input = tgt[:, :-1]  # 去掉最后一个元素
        tgt_output = tgt[:, 1:]  # 去掉第一个元素作为标签

        # 前向传播
        output = model(src.to(device), tgt_input.to(device))

        # 计算损失
        output = output.reshape(-1, output_dim)  # Flatten the output to (batch_size * seq_len-1, output_dim)
        tgt_output = tgt_output.contiguous().view(-1)  # Flatten the target to (batch_size * seq_len-1)

        loss = criterion(output, tgt_output.long().to(device))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.to(device).backward()
        optimizer.step()
        losshis[epoch] = loss.cpu().item()
        loss_stop = loss.cpu().item()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.cpu().item():.4f}')

    torch.save({'model': model, 'losshis': losshis}, 'model.pth')
    plt.plot(losshis)
    plt.ylim(0, 0.3)
    plt.draw()
    my_save_fig('./fig/losshis')
    return stop_epoch


def generalizing(task, input_dim, output_dim, cueUsed=None):
    print('generalize.')
    p = torch.load('model.pth')
    model = p['model']
    return training(task, input_dim, output_dim, model=model, shuffle_embed=True, cueUsed=cueUsed)


def generalize_ca3(task, input_dim, output_dim):
    p = torch.load('model.pth')
    model = p['model']
    return training(task, input_dim, output_dim, model=model, shuffle_ca3=True)


if __name__ == '__main__':
    # task = 'evidence'
    task = 'jiajian'

    delete_files_in_directory('./fig')

    # 假设输入序列长度为 10，词汇表大小为 5（0, 1, 2, 3, 4），批量大小为 32
    batch_size = 32
    seq_len = 100
    using_dim = 3 if task == 'evidence' else 2
    # input_dim = 3 if task == 'evidence' else 2  # 标准的embed层，限制维度
    input_dim = 30  # 不标准的embed层维度，用于进行后续的泛化
    output_dim = input_dim  # 因为目标序列是整数，输出是一个整数

    runnum = 5
    gennum = 2

    all_stop_epoch = np.zeros((runnum, gennum+1))
    for run in range(runnum):
        stop_epoch = training(task, input_dim, output_dim, batch_size, seq_len, cueUsed=0)  # 首次训练
        print(f'stop_epoch: {stop_epoch}')
        all_stop_epoch[run, 0] = stop_epoch
        for gen in range(gennum):
            # 泛化EC3
            # stop_epoch = generalizing(task, input_dim, output_dim, cueUsed=(gen+1)*using_dim)
            # 泛化CA3（改变位置编码）
            stop_epoch = generalize_ca3(task, input_dim, output_dim)

            print(f'stop_epoch: {stop_epoch}')
            all_stop_epoch[run, gen + 1] = stop_epoch

    np.savetxt("all_stop_epoch.csv", all_stop_epoch, delimiter=",")  # 保存成csv文件，(runnum, gennum)

    '''绘制 stop_epoch 结果'''
    plt.close()
    all_temp = np.zeros(gennum + 1)
    for runtime in range(runnum):
        temp = list(range(gennum + 1))
        for i in range(gennum + 1):
            temp[i] = all_stop_epoch[runtime, i]
        plt.scatter(list(range(gennum + 1)), temp)
        plt.plot(list(range(gennum + 1)), temp, color='black', linewidth=1)
        all_temp += np.array(temp)
    all_temp = all_temp / runnum
    plt.plot(list(range(gennum + 1)), all_temp, color='red', linewidth=3)
    plt.draw()
    plt.savefig('all_stop_epoch.jpg')
