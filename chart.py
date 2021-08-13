"""
    绘制蜡烛图
"""

import datetime as dt
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils import json2dict


def candle(data_x, data_y, title, x_label, y_label, save_path=None):
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

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_candle():
    data_file = 'result/main/face/mae.json'
    img_save_path = Path(data_file).parent.joinpath('pic.png').__str__()
    data = json.load(open(data_file, encoding='utf-8'))
    data_x = data['qualities']
    data_y = data['maes']
    candle(data_x, data_y, title='Error Compress K Line',
           x_label='quality', y_label='mae', save_path=img_save_path)


def plot_at():
    data_file = 'result/cv2_res/metrics.json'
    data = json2dict(data_file)
    mm = np.zeros((len(data), 2))
    for i, v in enumerate(data):
        acc = v['acc(20)']
        time = v['average time(ms)']
        mm[i, 0] = acc
        mm[i, 1] = time
    plt.xlabel('time')
    plt.ylabel('acc')
    plt.plot(mm[:, 1], mm[:, 0])
    plt.show()
    plt.savefig(Path(data_file).parent.joinpath('metrics.png').__str__())


if __name__ == '__main__':
    plot_at()
