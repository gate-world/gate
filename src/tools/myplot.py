import warnings

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from src.tools import DecoderCS
from src.lick.evidence_settings import evidence_stimu_num, evidence_stimu_duration
import os
from src.tools.delete_file import delete_files_in_directory
from src.tools.csv_process import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def delete_box():
    """去除plt的右侧和上侧边框"""
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def my_save_fig(base_filename):
    """
    防止对已经存在的图片进行覆盖
    :param base_filename: 不包含后缀的文件名。目前只处理jpg
    """
    filename = base_filename
    num = 1
    while os.path.exists(f"{filename}.png"):
        filename = f"{base_filename}_{num}"
        num += 1
    plt.savefig(f"{filename}.png", dpi=400)
    plt.close()


def heatmap(mat, isShow=True, isSave=None, vmax=None, vmin=None):
    """
    将numpy矩阵（二维）绘制为heatmap，colormap='heat'

    :param vmin: colorbar的最大最小值
    :param vmax:
    :param mat:
    :param isSave: 如果需要保存图片，则在这里填写保存路径（在外部调用plt.savefig无法完成保存）
    :return:
    """
    plt.close()
    data = pd.DataFrame(mat)
    fig = plt.figure(dpi=400, figsize=(6, 6))
    plot = sns.heatmap(data, vmax=vmax, vmin=vmin, cmap='viridis', cbar=False)  # 关闭colorbar

    # 设置热图x轴，y轴的ticks
    plt.xticks(rotation=0, fontsize=16)  # 设置x轴表明文字的放向，0为平放90为竖着放，注意一定要设置这个，否则无法正常显示！
    plt.yticks(rotation=0, fontsize=16)  #
    ax = plt.gca()
    xmajorLocator = MultipleLocator(25)  # 将x轴次刻度标签设置为25的倍数
    ax.xaxis.set_major_locator(xmajorLocator)
    xmajorFormatter = FormatStrFormatter('%d')  # 设置x轴标签文本的格式
    ax.xaxis.set_major_formatter(xmajorFormatter)
    '''用于普通的heatmap绘制'''
    ymajorLocator = MultipleLocator(50 if np.array(mat).shape[0] > 100 else 10)
    ax.yaxis.set_major_locator(ymajorLocator)
    ymajorFormatter = FormatStrFormatter('%d')  # 设置x轴标签文本的格式
    ax.yaxis.set_major_formatter(ymajorFormatter)
    '''用于Evidence细胞的绘制，强制设置 x 和 y 轴的刻度位置和标签'''
    # y_ticks = [12, 17, 22, 7, 2]  # 指定 y 轴刻度的位置
    # y_labels = ['0', '5', '10', '-5', '-10']  # 指定 y 轴刻度的标签
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels(y_labels, fontsize=16)  # 设置字体大小

    '''给Trace cell专用的细胞'''
    # x_ticks = [0, 10, 20, 30, 40, 50]  # 指定 y 轴刻度的位置
    # x_labels = ['-10', '0', '10', '20', '30', '40']  # 指定 y 轴刻度的标签
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels, fontsize=16)  # 设置字体大小

    if isSave is not None:
        plt.savefig(isSave + '.png', dpi=400)
        save_matrix_to_csv(mat, isSave + '.csv')
    if isShow:
        plt.show()  # 必须是save之后show，才能进行有效的图片保存
    plt.close()


def lick_plot(actionhis, cuehis, cueNum):
    """
    :param actionhis: (epochNum, bs, tracklength)
    :param cuehis: (epochNum, bs)
    """
    epochNum, bs, tracklength = actionhis.shape
    isLickGraph = np.zeros([cueNum, epochNum, tracklength])
    for epoch in range(epochNum):
        for cueType in range(cueNum):
            for sample in range(bs):
                if cuehis[epoch, sample] == cueType:
                    isLickGraph[cueType, epoch, :] = actionhis[epoch, sample, :]
                    break
    for cueType in range(cueNum):
        plt.clf()
        heatmap(isLickGraph[cueType, :, :], isShow=False, isSave=('./fig_result/_lick_%d' % cueType))


def sorting_plot(ca1_rate, save_name='', given_scale=None, given_sort=None):
    """
    用sorting的方式对细胞位置场进行排序，并绘制
    注意，scale会让每个细胞在绘制几张图的过程中，最大处为1
    used by: cs.py

    :param given_scale: 根据给定的scale值进行放缩，否则根据最大值放缩
    :param save_name: 保存图片时的图片名称
    :param given_sort: 是否给定细胞排序。如果给定，则不会进行排序
    :param ca1_rate: np.array, (x, cell_index)
    :return:
    """
    ca1_max_rate = np.amax(ca1_rate, axis=0)  # (cellNum,)
    ca1_max_rate_location = np.argmax(ca1_rate, axis=0)
    if given_sort is None:
        sorted_index = np.argsort(ca1_max_rate_location)  # 都是用cue=0的sorting index
    else:
        sorted_index = given_sort
    if given_scale is not None:
        assert(len(given_scale) == ca1_rate.shape[1])  # 确认细胞数量一致
        ca1_rate = ca1_rate / given_scale[np.newaxis, :]  # 添加一个维度以满足broadcast条件
    # else:
    #     ca1_rate = ca1_rate / ca1_max_rate[np.newaxis, :]
    sorted_image = ca1_rate[:, sorted_index].transpose().squeeze()
    heatmap(sorted_image, vmax=None if given_scale is None else 1.0,
            isShow=False, isSave=('./fig_result/_sorted_%s' % save_name))
    plt.clf()
    return sorted_index, ca1_max_rate


def sorting_plot_lap(unsilence_cell_mean, max_rate_each_cell):
    """
    只限于lap任务，但似乎效果不是很好
    :param max_rate_each_cell:
    :param unsilence_cell_mean: lapnum, x, cell
    :return:
    """
    # 拼接矩阵，让轨道拼接在一起。
    lapnum, tracklength, cellnum = unsilence_cell_mean.shape
    temp = np.reshape(unsilence_cell_mean, (lapnum*tracklength, cellnum))
    sorting_plot(temp, given_scale=max_rate_each_cell)


def trace_plot(unsilence_cell_mean):
    """
    只限于envb任务
    :param unsilence_cell_mean:
    :return:
    """
    lapnum, tracklength, cellnum = unsilence_cell_mean.shape
    fig = np.zeros((50, cellnum))
    for i in range(10):
        cueLocation = i * 5 + 10
        fig += unsilence_cell_mean[i, cueLocation-10:cueLocation+40, :]
    fig /= 10
    rows_to_delete = np.where(np.max(fig, axis=0) <= 0.1)  # 找到满足条件（最大值小于等于1）的列索引
    fig = np.delete(fig, rows_to_delete, axis=1)  # 删除这些列
    row_maxes = np.max(fig, axis=0, keepdims=True)
    fig /= (row_maxes + 0.01)
    sorting_plot(fig, save_name='envb')


def pf_corr_plot(cell_mean, isShuffle=False):
    """
    计算并绘制繁华前后，PF发放率的corr。
    :param cell_mean: (cueNum, track, cell)
    """
    from scipy.stats import pearsonr
    if os.path.exists('./fig_result/cell_mean.npz'):
        last_cell_mean = np.load('./fig_result/cell_mean.npz')['arr_0']
    else:
        return None
    assert (last_cell_mean.shape == cell_mean.shape)
    all_corr = []
    for i in range(last_cell_mean.shape[2]):  # cell num
        if np.max(last_cell_mean[:, :, i]) < 0.5 or np.max(cell_mean[:, :, i]) < 0.5:
            continue  # 对于发放率过低的细胞直接忽略
        corr, _ = pearsonr(np.mean(last_cell_mean[:, :, i], axis=0),
                           np.mean(cell_mean[:, :, i], axis=0))
        all_corr.append(corr)
    plt.close()
    plt.plot(all_corr)
    if isShuffle:
        plt.savefig('./fig_result/pf_corr_shuffle.png')
        save_matrix_to_csv(np.array(all_corr), './fig_result/pf_corr_shuffle.csv')
    else:
        plt.savefig('./fig_result/pf_corr.png')
        save_matrix_to_csv(np.array(all_corr), './fig_result/pf_corr.csv')
    return all_corr


def splitness_plot(ca1_max_rates):
    """
    见 sorting_plot。
    :param ca1_max_rates: np array, (cueNum, unsilence_cellNum)
    :return: np array, (cellNum,)
    """
    plt.close()
    std = np.std(ca1_max_rates, axis=0)  # 求列向量（每个细胞在不同cue时的标准差）
    this_splitness = std / (np.mean(ca1_max_rates, axis=0) + 0.5)  # 加上0.5是为了让那些发放率很低的细胞的splitness也很低
    # 如果有历史splitness强度信息的话，就一起绘制
    if os.path.exists('./fig_result/splitness.npz'):
        D = np.load('./fig_result/splitness.npz')
        splitness_his = D['arr_0']
        all_splitness = np.vstack((splitness_his, this_splitness))  # (n, cellNum)

        # 绘制倒数第二次和这一次的Splitness散点图
        plt.scatter(all_splitness[-2, :], all_splitness[-1, :])
        plt.xlabel('last splitness')
        plt.ylabel('this splitness')
        plt.plot((0, 0.1), (0, 0.1), color='r')  # x=y
        plt.axis('equal')
        my_save_fig('./fig_result/_splitness_scatter')
        delete_box()
        plt.close()

        # 绘制每个splitness-index curve
        for i in range(all_splitness.shape[0]):
            plt.plot(all_splitness[i, :], label=str(i))
        plt.legend()
        save_matrix_to_csv(all_splitness[-2:, :], './fig_result/_splitness.csv')
    else:
        plt.plot(this_splitness)
        all_splitness = this_splitness
    np.savez('./fig_result/splitness.npz', all_splitness)
    plt.grid()
    plt.savefig('./fig_result/_splitness.jpg')
    plt.close()
    return this_splitness


def evidence_plot(cue_left_right, eval_his):
    """
    :param cue_left_right: (bs, stimu_num), int, -1, 0, 1
    :param eval_his: (eval_bs, x, cell)
    :return:
    """
    os.makedirs('./fig_result/evidence', exist_ok=True)  # 保证目录的存在
    delete_files_in_directory('./fig_result/evidence')  # 清空文件夹

    last_bs, tracklength, cellNum = eval_his.shape
    stimu_num = evidence_stimu_num
    stimu_duration = evidence_stimu_duration
    cueLocation = (1 + np.array(range(stimu_num))) * stimu_duration * 2  # 每次刺激的起点位置，(stimu_num)
    cueDim = 2 * stimu_num + 1  # Evidence维度的binnum

    for cell in range(cellNum):
        rate_graph = np.zeros((cueDim, tracklength))
        visit_graph = np.zeros((cueDim, tracklength))
        for sample in range(last_bs):
            current_cue_location = 0
            current_evidence = stimu_num  # 为了绘制方便所以这样设置初始位置
            for x in range(tracklength):
                if x == cueLocation[current_cue_location]:
                    current_evidence += cue_left_right[sample, current_cue_location]  # 上下移动或者不动
                    if current_cue_location < stimu_num-1:
                        current_cue_location += 1
                visit_graph[int(current_evidence), x] += 1
                rate_graph[int(current_evidence), x] += eval_his[sample, x, cell]
        rate_graph = rate_graph / (visit_graph + 0.01)

        if np.max(rate_graph) > 0.1:  # 只绘制足够发放率足够大的细胞
            plt.close()
            heatmap(rate_graph, isShow=False, isSave='./fig_result/evidence/%d' % cell)


def cellular_plot(cell_indices, cell_mean):
    for cell_index in cell_indices:
        temp = cell_mean[:, :, cell_index]
        plt.close()
        cueNum = len(temp)
        for i in range(cueNum):
            plt.plot(temp[i, :], label=str(i), linewidth=2)
        plt.legend()
        delete_box()
        plt.draw()
        plt.savefig('./fig_result/_split%d.jpg' % cell_index)
        plt.close()
        save_matrix_to_csv(temp, './fig_result/_split%d.csv' % cell_index)


def AvsB_scatter(ca1_max_rate):
    """
    只绘制前两个cue类型之间的对比，用来判定是普通PF还是Splitter
    """
    plt.scatter(ca1_max_rate[0, :], ca1_max_rate[1, :])
    max_rate = np.max(ca1_max_rate)
    plt.plot([0, max_rate], [0, max_rate], color='red', linewidth=2)
    plt.gca().set_aspect('equal')
    delete_box()
    plt.draw()
    plt.savefig('./fig_result/_AvsB.png', dpi=400)
    save_matrix_to_csv(ca1_max_rate, './fig_result/_AvsB.csv')
    plt.close()


def evidence_poisson_plot(cue_left_right, eval_his):
    """
    :param cue_left_right: (bs, stimu_num), int, -1, 0, 1
    :param eval_his: (eval_bs, x, cell)
    :return:
    """
    os.makedirs('./fig_result/evidence', exist_ok=True)  # 保证目录的存在
    delete_files_in_directory('./fig_result/evidence')  # 清空文件夹

    last_bs, tracklength, cellNum = eval_his.shape
    one_way_max = 12  # 每个方向上统计的最大binnum
    cueDimMax = 2 * one_way_max + 1  # Evidence维度的binnum

    for cell in range(cellNum):
        rate_graph = np.zeros((cueDimMax, tracklength))
        visit_graph = np.zeros((cueDimMax, tracklength))
        for sample in range(last_bs):
            current_cue_location = 0
            current_evidence = one_way_max  # 为了绘制方便所以这样设置初始位置
            for x in range(tracklength):
                if current_evidence < cueDimMax-2 and cue_left_right[sample, x] == 1:
                    current_evidence += 1  # 上下移动或者不动
                if current_evidence > 0 and cue_left_right[sample, x] == -1:
                    current_evidence -= 1
                visit_graph[int(current_evidence), x] += 1
                rate_graph[int(current_evidence), x] += eval_his[sample, x, cell]
        rate_graph = rate_graph / (visit_graph + 0.01)

        if np.max(rate_graph) > 0.1:  # 只绘制足够发放率足够大的细胞
            plt.close()
            heatmap(rate_graph, isShow=False, isSave='./fig_result/evidence/%d' % cell)


def mds_plot(X, y, class_num):
    """

    :param X: (n, dim)
    :param y: (n,)
    :param class_num: 多少个样本类别，int
    """
    from src.tools.mds import MdsAnalyzer
    mds_a = MdsAnalyzer(n_components=2)
    x_embed = mds_a.fit(X)
    mds_a.plot(y, [str(i) for i in range(class_num)])
    save_matrix_to_csv(x_embed, './fig_result/_mds.csv')


def umap_plot(X, y):
    """
    用 UMAP 对每个时刻的细胞表征进行降维，看看能否出现【去相关】中的效果。
    效果很不错！dorsal的Split不算太明显，但是Inter和Ventral的 Split效果非常好
    :param X:
    :param y:
    :return:
    """
    from src.tools.my_umap import UmapAnalyzer
    umap = UmapAnalyzer()
    umap.fit(X)
    umap.plot(y)


def viewing(task, cueNum, train_his, actionhis, cueUsedhis, eval_his, cueUsed_eval, cue_left_right=None,
            is_plot_each_cell=False, is_scale_each_cell=False, is_silence_detect=False, isMDS=False):
    """
    绘制eval时的accuracy-location分析、
    PF-max位置分布图、
    PF在不同cue下的sorting图、
    lick行为随着时间演化的情况
    每个细胞随着epoch改变的heatmap

    :param is_silence_detect: 是否那些静息的细胞在sorting图中就不绘制了。在绘制Splitness时要使用False！
    :param is_scale_each_cell: 是否将所有的细胞发放率最大值调整为1。适合于 Lap 任务CA1的绘制；不适合EC5的绘制（因为有的小于0)
    :param isMDS: 是否用MDS对表征进行展示，可能运行会非常慢，但可视化效果很好
    :param is_plot_each_cell: 是否绘制每个细胞随着epoch变化的发放率。默认还是不绘制了，，，毕竟太费时间
    :param cue_left_right: 在Evidence 绘制的时候会用到，线索是否在左边。(eval_bs, stimu_num), int
    :param task: str: jiajian, decorrelate, lap, 1234
    :param cueNum: int; 注意在sequence任务中，cueNum并不等于actnum！！要单独拎出来讨论，否则cell_mean会有NaN
    :param train_his: (epoch, x, cell)
    :param actionhis: (epoch, bs), bool：isLick
    :param cueUsedhis: (epoch, bs)
    :param eval_his: (eval_bs, x, cell)
    :param cueUsed_eval: (eval_bs), int
    :return: None
    """
    os.makedirs('./fig_result', exist_ok=True)  # 保证目录的存在
    plt.close()

    arranged_batch = {}
    last_bs, tracklength, cellNum = eval_his.shape

    if task == 'sequence':  # 如果是Sequence任务，则无法进行cell_mean计算，因此只画最基本的图像
        statisticNum = 2
    else:
        statisticNum = cueNum  # 对多少种行为进行统计; 除了Sequence任务以外，其他任务都是二者相等

    cell_mean = np.zeros([statisticNum, tracklength, cellNum])  # 每个细胞在不同cue处的平均响应
    for cueType in range(statisticNum):
        arranged_batch[str(cueType)] = eval_his[cueUsed_eval == cueType, :, :]  # sample, x, cell
        cell_mean[cueType, :, :] = np.mean(arranged_batch[str(cueType)], axis=0)

    '''如果之前存储了cell_mean, 则进行繁华前后的pf corr 计算'''
    # pf_corr_plot(cell_mean)
    # np.savez('./fig_result/cell_mean.npz', cell_mean)  # 保存，以便下次繁华时使用
    # # 进行shuffle
    # cell_mean_shuffle = cell_mean.copy()
    # for j in range(cell_mean.shape[0]):
    #     np.random.shuffle(cell_mean_shuffle[j])
    # pf_corr_plot(cell_mean_shuffle, isShuffle=True)  # 对shuffle后结果进行计算

    '''绘制mds分析，以展示不同cueUsed表征之间的相似性'''
    if isMDS:  # 将表征对所有位置展平，运算可能比较慢
        mds_plot(eval_his.reshape(last_bs, tracklength*cellNum), cueUsed_eval, statisticNum)
        print('MDS done.')

    '''进行accuracy分析'''
    # 先提取每个bin的数据
    binNum = int(tracklength / 10)
    accuracy_list = np.zeros([binNum])
    train_test_ratio = 0.8
    for bin in range(binNum):  # 以0.1为一个区间，划分10个bin
        train_all_data = []
        train_all_label = []
        test_all_data = []
        test_all_label = []

        for cueType in range(statisticNum):
            temp = arranged_batch[str(cueType)][:,
                   int(10 * bin):int(10 * (bin + 1)), :]  # 每一段的x，用于CS+-分类
            sampleNum, binlength, cellNum = temp.shape
            train_sampple_size = int(train_test_ratio * sampleNum)
            shuffled_index = np.random.permutation(sampleNum)
            temp = temp[shuffled_index, :, :]  # 打乱顺序
            x_split = np.split(temp, (train_sampple_size,), axis=0)
            train_data = x_split[0]
            test_data = x_split[1]
            train_data = train_data.reshape(train_sampple_size * binlength, cellNum)
            test_data = test_data.reshape((sampleNum - train_sampple_size) * binlength, cellNum)

            label = cueType * np.ones([sampleNum * binlength, 1])
            y_split = np.vsplit(label, (train_sampple_size * binlength,))
            train_label = y_split[0]
            test_label = y_split[1]

            if cueType == 0:
                train_all_data = train_data
                train_all_label = train_label
                test_all_data = test_data
                test_all_label = test_label
            else:
                train_all_data = np.vstack((train_all_data, train_data))
                test_all_data = np.vstack((test_all_data, test_data))
                train_all_label = np.vstack((train_all_label, train_label))
                test_all_label = np.vstack((test_all_label, test_label))
        train_all_label = train_all_label.ravel()  # 展平成为(n,)形式
        test_all_label = test_all_label.ravel()

        # 在进行分类器训练和预测
        classifier = DecoderCS.BinaryClassifier()
        classifier.dataSetting(train_all_data, train_all_label, test_all_data, test_all_label)
        accuracy_list[bin] = classifier.dataTest()  # 每一个bin的分类准确率
    save_matrix_to_csv(accuracy_list.T, './fig_result/_decoding.csv')
    plt.close()
    plt.plot(0.05 + np.array(range(binNum)) * 0.1, accuracy_list)
    plt.title(f"Decoding Cue, {task}")
    plt.ylim(0, 1.1)
    plt.xlabel('time/s')
    plt.ylabel('accuracy')
    plt.savefig('./fig_result/_decoding.jpg')

    '''绘制最后一个epoch的最大发放率分布，eg：用来检查CA1是否过分稀疏，以绘制Histogram的方式'''
    plt.close()
    last_cell_rate = train_his[-1, :, :]  # x, index
    last_ca1_max_rate = np.amax(last_cell_rate, axis=0)
    silence_cell_threshold = 0.1  # 静息细胞的判定条件
    print('silence cell num: % d' % np.sum(last_ca1_max_rate < silence_cell_threshold))  # 静息细胞的数量，
    last_cell_rate = last_cell_rate[:, last_ca1_max_rate > silence_cell_threshold]  # 删掉最大发放率太低的那些静息细胞
    last_ca1_max_rate_location = np.argmax(last_cell_rate, axis=0)
    plt.hist(last_ca1_max_rate_location, bins=30)
    plt.title('last Epoch, distribution of Cell peak location')
    for cueType in range(statisticNum):
        mean_rate = np.mean(cell_mean[cueType, :, :], axis=1)
        plt.plot(mean_rate * 10, label=str(cueType))  # 绘制整体的平均发放率，看看EC3的平均发放率是否与CA1的接近
    plt.legend()
    plt.savefig('./fig_result/_distrib of peak.jpg', )

    '''绘制sorting图，以第一个cue的max_pos作为后续cue的排序'''
    plt.close()
    max_rate_each_cell = np.max(cell_mean.reshape((-1, cellNum)), axis=0)  # 每个细胞在所有cue下的最大发放率，(cell,)
    abs_rate_each_cell = np.max(np.abs(cell_mean).reshape((-1, cellNum)), axis=0)  # 如果不进行scale，就只筛选出abs足够大的
    if is_scale_each_cell:
        if any(max_rate_each_cell<0):
            warnings.warn('The Cell\'s max rate is smaller than 0! these cells will not be plotted.')
        unsilence_cell_mean = cell_mean[:, :, max_rate_each_cell > 0.05]
        max_rate_each_cell = max_rate_each_cell[max_rate_each_cell > 0.05]
    elif is_silence_detect:
        unsilence_cell_mean = cell_mean[:, :, abs_rate_each_cell > 0.05]
    else:
        unsilence_cell_mean = cell_mean
    ca1_max_rates = np.zeros((cueNum, unsilence_cell_mean.shape[2]))  # 用于绘制split 强度曲线
    sorted_index_cue_0 = None
    for cueType in range(statisticNum):
        last_cell_rate = unsilence_cell_mean[cueType, :, :]  # x, index

        if cueType == 0:  # 用cueType=0的细胞的排序，作为之后所有cueType的排序
            sorted_index_cue_0, ca1_max_rates[cueType, :] = sorting_plot(
                last_cell_rate, str(cueType), given_scale=max_rate_each_cell if is_scale_each_cell else None)
        else:
            _, ca1_max_rates[cueType, :] = sorting_plot(last_cell_rate, str(cueType), given_sort=sorted_index_cue_0,
                         given_scale=max_rate_each_cell if is_scale_each_cell else None)

    if task == 'lap':
        sorting_plot_lap(unsilence_cell_mean, max_rate_each_cell)
    if task == 'envb':
        trace_plot(unsilence_cell_mean)

    '''绘制能够区分普通PF与Splitter的scatter'''
    AvsB_scatter(ca1_max_rates)

    '''绘制每个细胞的Split强度曲线'''
    if not is_silence_detect and not is_scale_each_cell:  # 确保不管细胞发放不发放都统计，这样才能在vstack时保持尺寸一致
        splitness = splitness_plot(ca1_max_rates)
        most_splitness_cell_index = np.argsort(splitness)[-5:][::-1]
        cellular_plot(most_splitness_cell_index, unsilence_cell_mean)

    '''绘制lick的变化过程'''
    lick_plot(actionhis, cueUsedhis, statisticNum if task != 'sequence' else 2)

    '''如果是Evidence任务，绘制Evidence的二维发放率图'''
    if task == 'evidence':
        evidence_plot(cue_left_right, eval_his)
    elif task == 'evidence_poisson':
        evidence_poisson_plot(cue_left_right, eval_his)

    '''绘制每个细胞的发放率情况, 并保存为图片'''
    if is_plot_each_cell:
        plt.close()
        for i in range(min(100, cellNum)):
            mat = train_his[:, :, i]  # epoch, x
            plt.title('cell %d, max=%.5f' % (i, np.amax(mat)))
            heatmap(mat, isShow=False, isSave='./fig_result/%d' % i)
            print('cell fig %d saved. ' % i)
