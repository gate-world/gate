from src.tools.csv_process import *
from src.tools.my_barplot import *
import matplotlib.pyplot as plt


'''schedule_losshis.pth to csv'''
import torch
import numpy as np
losshis = torch.load('../DV/schedule_losshis.pth')
epoch_len = np.zeros([30, 4])
trial = 0
for loss in losshis:
    for i in range(4):
        epoch_len[trial, i] = len(loss[str(i)])
    trial += 1
save_matrix_to_csv(epoch_len, 'schedule_losshis.csv')
epoch_num = read_csv_to_matrix('schedule_losshis.csv')

# colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']  # 很难看的配色
'''plot'''
plot_significance_comparison(epoch_num.T,
                                 colors=np.array([[114, 170, 207], [254, 251, 186], [253, 185, 107], [236, 93, 59]])/255,
                                 labels=['training', '1st gen', '2nd gen', '3rd gen'])
ax = plt.gca()
# ax.set_ylabel('Epoch num', fontsize=16)
ax.set_yticks(np.linspace(100, 400, 4))

plt.draw()
plt.savefig('generalize_epochnum.png', dpi=800)
plt.show()