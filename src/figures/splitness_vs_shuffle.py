"""
绘制在ec3泛化中，Decorrelate任务，dorsal，intermediate和shuffle之间的splitness corr对比。
希望iCA1比dCA1的splitness corr更高，且两个都比shuffle高。
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from src.tools.csv_process import *
from src.tools.my_barplot import *

'''绘制和shuffle的对比'''
# each_gen_corr_shuffle = np.zeros((30, 1))
# each_gen_corr = np.zeros((30, 1))
# for i in range(30):
#     splitness = np.load('splitness/splitness_gen%d.npz' % i)['arr_0']  # 4, 100
#     for j in range(3):
#         corr, _ = pearsonr(splitness[j], splitness[j+1])
#         each_gen_corr[i] = corr
#         corr, _ = pearsonr(splitness[j], np.random.permutation(splitness[j+1].tolist()))
#         each_gen_corr_shuffle[i] = corr
# temp = np.hstack([each_gen_corr, each_gen_corr_shuffle])
# save_matrix_to_csv(temp, 'inter_splitness_corr_vs_shuffle_30X2.csv')  # 因为目前的数据是关于dorsal的splitness
# plot_significance_comparison(temp.T, colors=['gold', 'lightgray'], labels=['generalize', 'shuffle'])
# plt.yticks([0, 0.4, 0.8])
# plt.savefig('splitness_gen_vs_shuffle.png', dpi=800)
# plt.show()

'''对dorsal和intter进行比较，但似乎结果并不显著'''
# dorsal = read_csv_to_matrix('dorsal_splitness_corr_vs_shuffle_30X2.csv')
# inter = read_csv_to_matrix('inter_splitness_corr_vs_shuffle_30X2.csv')
# plot_significance_comparison(np.hstack((dorsal, inter)).T, labels=['dorsal_gen', 'shuffle', 'inter_gen', 'shuffle'])
# plt.show()

'''使用all_corr_during_gen来比较 dorsal 和 inter的差异'''
dorsal = read_csv_to_matrix('csvs/dorsal_all_corr_during_gen.csv')
inter = read_csv_to_matrix('csvs/inter_all_corr_during_gen.csv')
plot_significance_comparison(np.vstack((dorsal[:, 0], inter[:, 0], dorsal[:, 1], inter[:, 1])),
                             colors=['aquamarine', 'plum', 'lightseagreen', 'darkorchid'],
                             labels=['dCA1 1st', 'iCA1 1st', 'dCA1 2nd', 'iCA1 2nd'])
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.savefig('splitness_corr_d_vs_i.png', dpi=400)
plt.show()
