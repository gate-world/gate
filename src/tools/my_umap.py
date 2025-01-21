"""
用 UMAP 对每个时刻的细胞表征进行降维，看看能否出现【去相关】中的效果。
效果很不错！dorsal的Split不算太明显，但是Inter和Ventral的效果非常好
24.10.23
"""

import umap  # pip install umap-learn
import matplotlib.pyplot as plt
import numpy as np


class UmapAnalyzer:
    def __init__(self, n_components=3):
        """

        :param n_components: 将数据降维到多少个维度，一般来说2个就足够了
        """
        self.reducer = umap.UMAP(n_components=n_components)
        self.X_embed = None

    def fit(self, X):
        """
        :param X: 输入数据，（n, dim)
        """
        self.X_embed = self.reducer.fit_transform(X)  # (n, n_components), 降维完成的结果
        return self.X_embed

    def plot(self, y, index=None, cmap='bwr', save_path=None):
        """
        可视化结果
        :param index: 需要特殊标注的X index. 注意应当是由 range这样的list构成，而不是bool的列表
        :param cmap: bwr: 鲜艳红蓝；cool：蓝-紫；rainbow：彩虹色。https://matplotlib.org/2.0.2/users/colormaps.html
        :param y: 数字标签，(n,), int, 0 ~ class_num-1
        :param label_names: (class_num,), 每个类别的名称, str
        :param save_path: 图片保存的位置
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(self.X_embed[:, 0], self.X_embed[:, 1], self.X_embed[:, 2], c=y, cmap=cmap)
        if index is not None:  # 特定标记点
            # ax.scatter(self.X_embed[index, 0], self.X_embed[index, 1], self.X_embed[index, 2], s=70, marker='s')
            ax.scatter(self.X_embed[index[0], 0], self.X_embed[index[0], 1], self.X_embed[index[0], 2], s=500, color='k', marker='x')
        # ax.mouse_init()  # 用于鼠标事件初始化，这样就可以直接拖动图像查看了
        # plt.colorbar(sc)
        ax.axis('off')
        # colorbar.remove()
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)


if __name__ == '__main__':

    '''iris数据集'''
    # from sklearn.datasets import load_iris
    # iris = load_iris()
    # X = iris.data  # 特征数据
    # y = iris.target  # 目标变量（类别标签）

    '''我的数据集，尝试画出【去相关】中的效果'''
    D = np.load('../DV/cells.npz')
    # ca1his: (epoch, loop, x, cell); cueUsedhis: (epoch, bs); actionhis: (epoch, bs)
    ca1his, cueUsedhis = D['arr_0'], D['arr_3']
    tracklength = ca1his.shape[2]
    Loop = 1
    epochs = list(range(100, 120))  # 临近训练结束时的epoch
    # epochs = list(range(ca1his.shape[0]-20, ca1his.shape[0]))
    epochnum = len(epochs)
    assert(ca1his.shape[0] > epochs[-1])  # 保证epoch没越界
    X = ca1his[epochs, Loop, :, :].reshape([epochnum*tracklength, -1])  # dorsal的 split不算太明显，但Intermediate就已经非常明显了
    cueTypes = cueUsedhis[epochs, 0]

    # y = np.repeat(list(range(20)), tracklength).astype(int)  # 按照epoch来给定标签，有20种class
    y = np.repeat(cueTypes, tracklength).astype(int)  # 按照线索类型给出标签，就两种class
    # 给没有线索刺激的那些样本点标注出来，并用红叉显示轨道起点
    index = []
    for i in range(epochnum):
        index += list(range(tracklength*i, tracklength*i+20))

    print('data loaded.')
    umap_analyzer = UmapAnalyzer()
    x_embed = umap_analyzer.fit(X)
    umap_analyzer.plot(y, index=index)
    from src.tools.csv_process import *
    np.savez('umap_result%d.npz' % Loop, x_embed)



