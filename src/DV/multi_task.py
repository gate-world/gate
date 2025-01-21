"""
尝试对多个任务进行（同步）学习
24。11.13
"""

from src.DV.dv_gating import *

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

task_list = ('decorrelate', 'lap', '1234')

loopnum = 3
epochNum = 50
batch_size = 128 if isGPU else 30  # 如果是GPU训练就可以使用很大的bs而不影响单个epoch的训练时间
lr = 0.01

runNum = 10

'''实际进行schedule运算，每轮都保存结果，并且保存环境信息（CA3排布）'''
loss, net = net_train(loopnum, task_list[-1], epochNum, batch_size, lr,
                      is_plot_block=False, single_act_point=False, device=device)
losshis = {'0': loss}
task_num = len(task_list)
given_ca3_order = [0]*task_num
p = torch.load('hpc_rnn.pth')
given_ca3_order[-1] = p['ca3order']
wca1act = [0]*task_num
wca1act[-1] = p['wca1act']
actbias = [0]*task_num
actbias[-1] = p['actbias']
cuePattern = [0]*task_num
cuePattern[-1] = p['cuePattern']

for runtime in range(runNum):
    task_index = runtime % len(task_list)
    task = task_list[task_index]
    print(f'task = {task}')
    if runtime < task_num - 1:  # 既改变CA3也改变EC3
        p = torch.load('hpc_rnn.pth')
        pre_loss_his = p['losshis']
        losshis[str(runtime + 1)], net = net_train(loopnum, task, epochNum, batch_size, lr,
                                                   given_net=net, isShuffleCA3=True,
                                                   pre_losshis=pre_loss_his, is_plot_block=False, device=device)
        p = torch.load('hpc_rnn.pth')
        given_ca3_order[task_index] = p['ca3order']
        cuePattern[task_index] = p['cuePattern']
    else:  # 使用原有的EC3，以及CA3，还有之前训练过的act权重
        p = torch.load('hpc_rnn.pth')
        pre_loss_his = p['losshis']
        losshis[str(runtime + 1)], net = net_train(loopnum, task, epochNum, batch_size, lr,
                                                   givenCA3order=given_ca3_order[task_index],
                                                   given_cuePattern=cuePattern[task_index],
                                                   given_wca1act=wca1act[task_index],
                                                   given_actbias=actbias[task_index],
                                                   pre_losshis=pre_loss_his,
                                                   given_net=net,
                                                   is_plot_block=False, device=device)
    p = torch.load('hpc_rnn.pth')
    actbias[task_index] = p['actbias']
    wca1act[task_index] = p['wca1act']

    plt.ylim(0, 1)
    plt.savefig(f'./fig_result/_loss result_{runtime+1}.jpg')
    plt.close()
    print(f'runtime {runtime+1} done.')
