"""
对比多次繁华中，Splitness的变化。希望是越来越高的。
"""
import numpy as np
from src.tools.my_barplot import *
from src.tools.csv_process import *

gen_corr = read_csv_to_matrix('../DV/all_corr_during_gen.csv')

plot_significance_comparison(gen_corr.T,
                                 colors=np.array([[254, 251, 186], [253, 185, 107], [236, 93, 59]])/255,
                                 labels=['1st gen', '2nd gen', '3rd gen'])
ax = plt.gca()
# ax.set_ylabel('CA1 activity correlation', fontsize=16)
ax.set_yticks([0.4, 0.8, 1.2])
ax.tick_params(axis='y', labelsize=16)

plt.draw()
plt.savefig('splitness_corr_during_gen.png', dpi=800)
plt.show()
