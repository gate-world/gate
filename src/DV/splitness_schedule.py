"""
用于验证泛化过程中，是否原有的Split细胞在泛化之后也是Split细胞。
主要观察splitness.jpg这张图就可以了。EC3泛化中的效果还是非常好的！

注意：
一定要将dv_gating.py中的viewing代码的默认参数 is_silence_detect=False 加上去！
否则在绘制Splitness曲线的时候会报错。
（默认的参数是True）

oneNote：241012

24。10.24
"""
import matplotlib.pyplot as plt

from src.DV.dv_gating import *
import numpy as np


myplot.delete_files_in_directory('./fig_result')    # 重新训练时先清空文件夹（用于保证绘制Splitness不出问题）

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
epochNum = 500
batch_size = 128 if isGPU else 30  # 如果是GPU训练就可以使用很大的bs而不影响单个epoch的训练时间
lr = 0.01
cueNum = cs.get_cue_num(task)  # 每一轮刚开始的时候都重置一下

genNum = 3  # 似乎泛化的次数越多，泛化效果就越好，不同实验之间的表征一致性会强很多

loop_to_plot = 0  # 选择是第几个loop进行绘制
cell_type_plot = 'ca1'  # 选择的是哪种细胞进行绘制
training(loopnum, task, epochNum, batch_size, lr, is_plot_block=False, single_act_point=False)
viewing(task, cueNum, loop_to_plot, cell_type_plot, is_plot_each_cell=False)
for i in range(genNum):
    generalizing(loopnum, task, epochNum, batch_size, lr, cueNum=cueNum,
                                 gen_type='change_ec3',
                                 is_plot_block=False, single_act_point=False)
    viewing(task, cueNum, loop_to_plot, cell_type_plot, is_plot_each_cell=False)

"""
再次提醒！

一定要将dv_gating.py中的viewing代码的默认参数 is_silence_detect=False 加上去！
否则在绘制Splitness曲线的时候会报错。
（默认的参数是True）
"""
