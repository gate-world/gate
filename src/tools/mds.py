"""
对数据进行MDS降维分析，以展示表征和表征之间的相似性
使用天工ai的代码。
24.10.14
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import MDS


class MdsAnalyzer:
    def __init__(self, n_components=2, random_state=1):
        """

        :param n_components: 将数据降维到多少个维度，一般来说2个就足够了
        """
        self.mds = MDS(n_components=n_components, random_state=random_state)
        self.X_mds = None

    def fit(self, X):
        """
        :param X: 输入数据，（n, dim)
        """
        self.X_mds = self.mds.fit_transform(X)  # (n, n_components), 降维完成的结果
        return self.X_mds

    def plot(self, y, label_names, save_path='./fig_result/_mds.jpg'):
        """
        可视化结果
        :param y: 数字标签，(n,), int, 0 ~ class_num-1
        :param label_names: (class_num,), 每个类别的名称, str
        :param save_path: 图片保存的位置
        """
        plt.figure(figsize=(6, 6))
        for i, label in enumerate(label_names):
            plt.scatter(self.X_mds[y == i, 0], self.X_mds[y == i, 1], label=label)
        plt.xticks([])  # 取消x刻度
        plt.yticks([])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.axis('square')
        # plt.legend()
        plt.draw()
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    # 加载数据集
    data = load_iris()
    X = data.data  # (150, 4)
    y = data.target  # (150,)

    # 创建MDS实例并降维
    mds_a = MdsAnalyzer()
    mds_a.fit(X)
    mds_a.plot(y, data.target_names)
