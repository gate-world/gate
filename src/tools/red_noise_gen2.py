"""
另一种红噪声的生成方式；用filtfilt滤波器来生成红噪声。
loglog图像显示出的结果与1000个EC3求和的结果一致，只是曲线不是非常好看。
（已废弃）改用pink_noise_gen
24.08.11
"""
import numpy as np
from scipy.signal import filtfilt, butter


# 定义函数生成红噪声
def generate_red_noise(length=1000, sample_rate=1.0):
    # 生成白噪声
    white_noise = np.random.randn(length)

    # 设定滤波器的参数
    nyquist_freq = 0.5 * sample_rate
    cutoff = 0.08 * nyquist_freq  # 这里设置的截止频率较低，以增加低频能量
    b, a = butter(1, cutoff / nyquist_freq, btype='low', analog=False)

    # 使用滤波器处理白噪声
    filtered_noise = filtfilt(b, a, white_noise)

    return filtered_noise


# 生成红噪声信号
signal_length = 10000
sample_rate = 10
red_noise = generate_red_noise(length=signal_length, sample_rate=sample_rate)
red_noise[0:2] = 0
red_noise[-3:-1] = 0
red_noise = red_noise / np.std(red_noise) * 7

# 可以使用matplotlib库进行可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(red_noise)
plt.title('Red Noise Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# 可视化功率谱密度
psd = np.abs(np.fft.fft(red_noise)) ** 2 / signal_length
frequencies = np.fft.fftfreq(len(psd), 1 / sample_rate)
plt.loglog(frequencies[:signal_length // 2], psd[:signal_length // 2])
plt.title('Power Spectral Density of Pink Noise')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density [V**2/Hz]')
plt.grid(True)
plt.show()
