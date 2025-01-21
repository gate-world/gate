"""
直接绘制colorbar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 生成一些示例数据
x = np.linspace(0, 10, 100)
y = 0.5*np.sin(x)+0.5

# 创建一个新的colormap，这里使用magma
# cmap = plt.cm.magma
cmap = plt.cm.viridis


# 绘制图表
fig, ax = plt.subplots()
sc = ax.scatter(x, y, c=y, cmap=cmap)

# 创建一个单独的colorbar
cbar = fig.colorbar(sc, fraction=0.9)
cbar.set_ticks([])
plt.savefig('colorbar.png', dpi=400)

# 显示图表
plt.show()
