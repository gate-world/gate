"""
绘制pf的corr，对比繁华与shuffle。希望看到繁华过程中保持位置场。
"""
import numpy as np
from src.tools.my_barplot import *
from src.tools.csv_process import *

gen_corr = read_csv_to_matrix('pf_corr.csv')
shuffle_corr = read_csv_to_matrix('pf_corr_shuffle.csv')
plot_significance_comparison([gen_corr, shuffle_corr],
                                 colors=['gold', 'lightgray'],
                                 labels=['generalize', 'shuffle'])
ax = plt.gca()
# ax.set_ylabel('CA1 activity correlation', fontsize=16)
ax.set_yticks(np.linspace(-0., 1.0, 3))

plt.draw()
plt.savefig('pv_corr_vs_shuffle.png', dpi=800)
plt.show()
