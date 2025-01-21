"""
用于绘制多次训练中的loss his曲线。
注意，只训练了一次的话losshis是一个list，会报错。
"""

from src.DV.dv_gating import *
from src.tools.myplot import my_save_fig


def plot_his_from_dict(dict):
    plt.rcParams['ytick.labelsize'] = 16  # 设置全局y轴刻度字体大小为16
    plt.rcParams['xtick.labelsize'] = 16

    for i, value in dict.items():
        if value is not None:
            plt.plot(value)
    ax = plt.gca()
    plt.ylim([0, 0.4])
    plt.yticks([0.2, 0.4])
    plt.xticks([50, 100, 150, 200])
    # 去除上侧和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.draw()
    my_save_fig('./loss_his_plot/loss')


def plot_thumbnail(losshis):
    """
    绘制 lap任务的时候专门使用的。放大 x轴来绘制最开始的结果
    :param losshis: dict, 'i'对应一个loss list
    :return:
    """
    plt.close()
    fig = plt.figure(figsize=[3, 3])
    for i in range(len(losshis)):
        plt.plot(losshis[str(i)])
    ax = plt.gca()
    plt.xlim([0,10])
    # 去除上侧和右侧边框
    plt.yticks([0.2, 0.4, 0.6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.draw()
    plt.savefig('lap_gen_loss_thumb.png', dpi=400)

    plt.show()


if __name__ == '__main__':
    # 绘制一次run的结果
    # losshis = torch.load('../DV/hpc_rnn.pth')['losshis']
    # plot_his_from_dict(losshis)

    # 绘制多次run的结果
    temp = torch.load('schedule_losshis.pth')  # runnum 个dict
    for i in range(len(temp)):
        plot_his_from_dict(temp[i])
