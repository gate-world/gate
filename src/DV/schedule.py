"""
按照计划顺序，多次执行dv_gating。py中的代码
例如，验证多次泛化的学习速度是否会提升
24。10.09
"""
import matplotlib.pyplot as plt
import os
from src.tools.csv_process import *
from scipy.stats import pearsonr
from src.DV.dv_gating import *
import numpy as np


seed = 2  # 设置全局的随机种子，用来确保实验可复现; lap 500的种子（服务器）
print(f'\n  ----Seed Using: {seed}')
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# 检查device情况
if torch.cuda.is_available():
    isGPU = True
    device = torch.device("cuda:0")
else:
    isGPU = False
    device = torch.device("cpu")
print('\n  -----------Device using: ---------')
print(device)

# task = 'jiajian'  # 任务类型, 注意jiajian任务的轨道长度为200而非100，会导致默认的准确率就很高
task = 'decorrelate'
# task = 'lap'
# task = 'alter'
# task = '1234'
# task = 'evidence'
# task = 'evidence_poisson'
# task = 'sequence'

loopnum = 3
epochNum = 300
batch_size = 64 if isGPU else 32  # 如果是GPU训练就可以使用很大的bs而不影响单个epoch的训练时间
lr = 0.01

runNum = 30
genNum = 3

compute_splitness = True

'''实际进行schedule运算，每轮都保存结果'''
losshis = [{'dd': None} for i in range(runNum)]
isTrainSuccess = [True]*runNum
all_corr = [[] for i in range(runNum)]
for runtime in range(runNum):
    # seed = runtime  # 每次运行使用不同的种子
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

    plt.close()
    cueNum = cs.get_cue_num(task)  # 每一轮刚开始的时候都重置一下
    losshis[runtime]['0'], _ = training(loopnum, task, epochNum, batch_size, lr,
                                     is_plot_block=False, single_act_point=False)
    if compute_splitness:
        viewing(task, cueNum, loop_to_plot=0, cell_type_plot='ca1', is_plot_each_cell=False)
    print(f'training runtime {runtime} done.')
    if losshis[runtime]['0'][-1] > 5:
        isTrainSuccess[runtime] = False
        continue  #
    for generalize_time in range(genNum):
        if task == 'sequence':
            cueNum += 1
        else:
            cueNum = cs.get_cue_num(task)
        lossh = generalizing(loopnum, task, epochNum, batch_size, lr, cueNum=cueNum,
                             gen_type='change_nothing' if task == 'sequence' else 'change_ec3',
                             is_plot_block=False, single_act_point=False)
        losshis[runtime][str(1+generalize_time)] = lossh

        plt.grid(False)
        plt.legend.remove()
        plt.ylim(0, 1)
        plt.savefig(f'./fig_result/_loss result_{runtime}.jpg')
        plt.close()

        if compute_splitness:
            viewing(task, cueNum, loop_to_plot=0, cell_type_plot='ca1', is_plot_each_cell=False)
        print(f'runtime {runtime}, generation {generalize_time} done.')

        # 计算splitness的correlate
        if compute_splitness:
            splits = read_csv_to_matrix('./fig_result/_splitness.csv')  # 2*n
            corr, _ = pearsonr(splits[0, :],
                               splits[1, :])
            all_corr[runtime].append(corr)

    torch.save(losshis, 'schedule_losshis.pth')
    if compute_splitness:
        os.rename('./fig_result/splitness.npz', './fig_result/splitness_gen%d.npz' % runtime)
if compute_splitness:
    all_corr = np.array(all_corr)
    save_matrix_to_csv(all_corr, 'all_corr_during_gen.csv')





'''绘制所有结果'''
losshis = torch.load('schedule_losshis.pth')
plt.close()
all_temp = np.zeros(genNum+1)
for runtime in range(runNum):
    if not isTrainSuccess[runtime]:
        all_temp += epochNum * np.ones(genNum+1)
        continue
    temp = list(range(genNum+1))
    for i in range(genNum+1):
        temp[i] = len(losshis[runtime][str(i)])
    plt.scatter(list(range(genNum+1)), temp)
    plt.plot(list(range(genNum+1)), temp, color='black', linewidth=1)
    all_temp += np.array(temp)
all_temp = all_temp / runNum
plt.plot(list(range(genNum+1)), all_temp, color='red', linewidth=3)
plt.draw()
plt.savefig('all_losshis')
