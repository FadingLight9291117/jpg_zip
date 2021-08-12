"""
    绘制蜡烛图
"""

import numpy as np
import datetime as dt
import matplotlib.pyplot as mp
import matplotlib.dates as md


def plot_candle(data):
    dates = [i + 1 for i in range(10)]
    min_mae = [1] * 10
    max_mae = [2, 3, 4, 5, 6, 7, 10, 2, 6, 1]

    k = 2.5
    mean_mae = [1.5] * 10
    std_mae = [0.5] * 10
    max_mae2 = np.array([i + j * k for i, j in zip(mean_mae, std_mae)])
    min_mae2 = np.array([i + j * k for i, j in zip(mean_mae, std_mae)])

    # 2.设置绘图窗口
    mp.figure("Apple K Line", facecolor="lightgray")
    mp.title("Apple K Line", fontsize=16)
    mp.xlabel("Data", fontsize=14)
    mp.ylabel("Price", fontsize=14)

    # 3.x坐标（时间轴）轴修改
    ax = mp.gca()
    # 设置主刻度定位器为周定位器（每周一显示主刻度文本）
    # ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
    # ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
    # ax.xaxis.set_minor_locator(md.DayLocator())

    mp.tick_params(labelsize=8)
    mp.grid(linestyle=":")

    # 4.判断收盘价与开盘价 确定蜡烛颜色
    colors_bool = max_mae2 >= min_mae2
    colors = np.zeros(colors_bool.size, dtype="U5")
    colors[:] = "blue"
    colors[colors_bool] = "white"

    # 5.确定蜡烛边框颜色
    edge_colors = np.zeros(colors_bool.size, dtype="U1")
    edge_colors[:] = "b"
    edge_colors[colors_bool] = "r"
    #
    # # 绘制开盘价折线图片
    # dates = dates.astype(md.datetime.datetime)
    # mp.plot(dates, open_price, color="b", linestyle="--",
    #         linewidth=2, label="open", alpha=0.3)
    #
    # # 6.绘制蜡烛
    mp.bar(dates, (max_mae2), 0.8, bottom=min_mae2, color=colors,
           edgecolor=edge_colors, zorder=3)

    # 7.绘制蜡烛直线(最高价与最低价)
    # mp.vlines(dates, min_price, max_price, color=edge_colors)

    mp.vlines(dates, min_mae, max_mae)
    mp.legend()
    mp.gcf().autofmt_xdate()
    mp.show()
