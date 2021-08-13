"""
    绘制蜡烛图
"""

import datetime as dt
from dataclasses import dataclass
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md


def plot_candle(data_x, data_y, title, x_label, y_label):
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    max_data = data_y.max(axis=1)
    min_data = data_y.min(axis=1)

    mean_data = data_y.mean(axis=1)
    std_data = data_y.std(axis=1)

    # 2.设置绘图窗口
    plt.figure(title, facecolor="lightgray")
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # 3.x坐标（时间轴）轴修改
    ax = plt.gca()

    plt.tick_params(labelsize=8)
    plt.grid(linestyle=":")

    # # 6.绘制蜡烛
    plt.bar(data_x, std_data, 1, bottom=mean_data - std_data / 2, color='red',
            edgecolor='black', zorder=3)

    # 7.绘制蜡烛直线(最高价与最低价)
    plt.vlines(data_x, min_data, max_data)

    # 8. 绘制均值曲线
    plt.plot(data_x, mean_data)
    # plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    data_file = 'result/error_main/face/mae.json'
    data = json.load(open(data_file, encoding='utf-8'))
    data_x = data['qualities']
    data_y = data['maes']
    plot_candle(data_x, data_y, title='Error Compress K Line',
                x_label='quality', y_label='mae')
