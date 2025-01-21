"""
gpt生成的柱状图，以及显著性检验（默认使用t-test）ttest_ind

其他显著性检验方法：

Mann-Whitney U 检验 (mannwhitneyu): 非参数检验，用于检验两个独立样本是否来自同一分布，适用于非正态数据。
u_stat, p_value = stats.mannwhitneyu(group1, group2)———————— 【dv轴】中使用的方法

ANOVA 单因素方差分析 (f_oneway): 用于检验三个或更多样本的均值是否相等。
f_stat, p_value = stats.f_oneway(group1, group2, group3)

Kolmogorov-Smirnov 检验 (ks_2samp): 检验两个样本是否来自相同的分布，适用于两组数据的比较。
ks_stat, p_value = stats.ks_2samp(group1, group2)


24.11.13
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from itertools import combinations


def plot_significance_comparison(data, colors=None, labels=None):
    """
    绘制柱状图并对比各个维度间的显著性，标记星号，显示显著性检验的横向线段，并添加散点图展示样本分布。

    参数：
    data: list of arrays，每个数组对应一个维度的数据。每个数组的长度可以不一样。
    colors: list of strings，指定每个维度的颜色。默认为 None，自动生成颜色。
    labels: legend中使用的名称，如果为None则不绘制 legend

    返回：
    None
    """

    # 确保输入的数据是有效的
    num_groups = len(data)
    for i in range(num_groups):
        if not isinstance(data[i], (list, np.ndarray)):
            raise ValueError(f"数据第{i + 1}维度的类型错误，必须是列表或数组。")

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # 如果没有指定颜色，则自动生成颜色
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, num_groups))

    # 计算每个维度的平均值
    means = [np.mean(group) for group in data]
    sems = [np.std(group) / len(group) for group in data]
    num_samples = [len(group) for group in data]

    # 绘制柱状图
    positions = np.arange(num_groups)
    bars = ax.bar(positions, means, yerr=sems, capsize=5, color=colors, edgecolor='black')
    # 为每个bar单独添加legend
    if labels is not None:
        assert(len(labels) == len(data))
        for i, rect in enumerate(bars):
            rect.set_label(f'{labels[i]}')
    # ax.legend(loc='best')

    # 添加抖动的散点图以展示每组的样本点分布
    jitter_strength = 0.2  # 调整抖动强度
    for i, group in enumerate(data):
        jittered_x = positions[i] + np.random.uniform(-jitter_strength, jitter_strength, len(group))
        ax.scatter(jittered_x, group, color=colors[i], alpha=0.6, edgecolor='black', linewidth=0.5)

    # 动态生成显著性检验的顺序
    # 生成所有可能的组对
    comparisons = list(combinations(range(num_groups), 2))

    # 排序：先检验相邻组，再检验远离的组
    comparisons.sort(key=lambda x: abs(x[0] - x[1]))  # 按照距离排序

    datamax = np.max([np.max(group) for group in data])

    # 计算并标注显著性
    y_offset_step = 0.1*datamax
    y_offset = 0.1*datamax  # 初始偏移量
    group_dist = 1
    eps_shift = 0.05  # 用于微小的便宜，让竖着的短线错开
    for i, j in comparisons:  # 永远都是i<j
        # 计算 t-test 显著性
        t_stat, p_value = stats.mannwhitneyu(data[i], data[j])

        # 根据 p-value 决定星号数量
        if p_value < 0.001:
            star = '***'
            color = 'black'
        elif p_value < 0.01:
            star = '**'
            color = 'black'
        elif p_value < 0.05:
            star = '*'
            color = 'black'
        else:
            star = 'n.s.'
            color = 'gray'

        # 每次绘制后，增加y_offset
        if abs(i - j) > group_dist:
            y_offset += y_offset_step  # 每次增加一定的偏移量

        # 找到两组的柱状图顶部位置
        y_pos = 1.1*datamax + y_offset

        # 在柱状图上方添加星号标记
        ax.text((i + j) / 2, y_pos-0.05*y_offset_step, star, ha='center', va='bottom', fontsize=16, color=color)

        # 绘制横向线段
        ax.plot([i+eps_shift, j-eps_shift], [y_pos, y_pos], color='black', lw=1)
        ax.plot([i+eps_shift, i+eps_shift], [y_pos - datamax*0.015, y_pos], color='black', lw=1)
        ax.plot([j-eps_shift, j-eps_shift], [y_pos - datamax*0.015, y_pos], color='black', lw=1)

    # 设置标题和标签
    # ax.set_title('Significance Comparison of Groups', fontsize=16)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # 去除上侧和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 显示图形
    plt.tight_layout()


if __name__ == '__main__':

    # 示例数据（不指定组数，函数根据输入数据的维度自动确定）
    data1 = np.random.normal(105, 2, 50)
    data2 = np.random.normal(125, 2, 60)
    data3 = np.random.normal(145, 2, 55)
    data4 = np.random.normal(125, 2, 60)

    # 调用函数，数据和颜色
    plot_significance_comparison([data1, data2, data3, data4],
                                 colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'],
                                 labels=['a', 'b', 'c', 'd'])
