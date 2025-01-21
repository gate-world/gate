"""
对loss进行即时平滑，每次更新loss时返回平滑的结果
24.10.16
"""
import numpy as np


class LossSmoother:
    def __init__(self, window_width=10):
        self.window_width = window_width
        self.his = np.ones(1)
        self.epoch = 0

    def update_loss(self, loss):
        self.his = np.hstack((self.his, np.array(loss)))
        self.epoch += 1
        if self.epoch == 1:
            return loss
        if self.epoch < self.window_width:  # epoch比滑动窗还小的话就不平滑了
            return (self.his[1]*(self.window_width-self.epoch) + np.sum(self.his)) / self.window_width
        else:
            return np.mean(self.his[-self.window_width:-1])

    def get_all_loss(self):
        return self.his
