"""
用滤波的方式，来生成红噪声
目前用于给EC3的指数发放率加上噪声，以模仿 EC3 Markov过程生成的信号
但问题是loglog中的斜率与EC3 Markov并不一样
24.08.10
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter, butter


def generate_pink_noise(length, fs=10):
    """根据长度，生成标准差为1，期望为0的红噪声"""
    # 生成白噪声
    white_noise = np.random.randn(length)

    b = [1, 0]  # 高通滤波器系数
    a = [1, -0.999]  # 高通滤波器系数
    sig = lfilter(b, a, white_noise) / np.std(white_noise)  # 标准的粉红噪声

    # 进一步限制低频部分的功率，让噪声更加贴近EC3的效果
    nyq = 0.5 * fs
    cutoff = 0.08
    normal_cutoff = cutoff / nyq
    b, a = butter(1, normal_cutoff, btype='high', analog=False)  # 阶数为1的效果比较好
    sig = lfilter(b, a, sig)
    return sig / np.std(sig)


if __name__ == '__main__':
    # 参数设置
    signal_length = 10000
    sample_rate = 10

    cell_num = 1000

    # 生成粉红噪声
    all_noise = np.zeros(signal_length)
    for _ in range(cell_num):
        pink_noise = generate_pink_noise(signal_length)
        pink_noise = pink_noise * 0.24 + 0.06  # 对标1000个EC3的方差以及期望
        all_noise += pink_noise

    # 绘制信号
    print(np.var(all_noise))
    plt.figure()
    plt.plot(np.linspace(0, signal_length / sample_rate, signal_length), all_noise)
    plt.ylim(0, 100)
    plt.show()

    # 可视化功率谱密度
    psd = np.abs(np.fft.fft(all_noise)) ** 2 / signal_length
    frequencies = np.fft.fftfreq(len(psd), 1 / sample_rate)
    plt.loglog(frequencies[:signal_length // 2], psd[:signal_length // 2])
    plt.title('Power Spectral Density of Pink Noise')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [V**2/Hz]')
    plt.grid(True)
    plt.show()
