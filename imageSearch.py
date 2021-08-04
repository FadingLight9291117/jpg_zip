"""
    穷举法搜索一张大图中的小图位置，使用最小MAE判别
"""
import json
import time
from pathlib import Path
import argparse
from typing import List, Dict
import copy

import cv2
import numpy as np
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

from utils import Timer

from searchMethod import fftSearch, spaceSearch


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imgL-path', type=str, default='./data/images_2021_07_29/1.jpg', help='大图的文件夹路径')
    parser.add_argument('--imgS-path', type=str, default='./data/images_2021_07_29/1_1.jpg', help='小图的文件夹路径')
    parser.add_argument('--result-path', type=str, default='./result', help='存放结果的文件夹')
    parser.add_argument('--search-method', type=str, default='fft', help='搜索方法')
    parser.add_argument('--show', type=int, default=1, help='是否显示结果图片')

    return parser.parse_args()


def main_search(imgL_file, imgS_file, search_method):
    imgL = cv2.imread(str(imgL_file))
    imgS = cv2.imread(str(imgS_file))

    if search_method == 'space':
        search_method = spaceSearch.spaceSearch
    elif search_method == 'fft':
        search_method = fftSearch.fftSearch

    box = search_method(imgL, imgS)

    result = {
        'imgL_path': imgL_file,
        'imgS_path': imgS_file,
        'box': box,
    }
    return result


def save_res(res, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)
        print(f'result save in "{save_path}"')


def show_res(**kargs):
    res = kargs.get('res')
    file = kargs.get('file')

    if res:
        imgL_path = res['imgL_path']
        imgS_path = res['imgS_path']
        box = res['box']
        box = edict(box)
    elif file:
        with open(file, encoding='utf-8') as f:
            data = json.load(f)
        data = edict(data)
        imgL_path = data.imgL_path
        imgS_path = data.imgS_path
        box = edict(data.result)
    else:
        return

    imgL = cv2.imread(imgL_path)
    imgS = cv2.imread(imgS_path)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    imgL = cv2.rectangle(imgL, (box.x1, box.y1), (box.x2, box.y2), color=(255, 0, 0), thickness=5)

    # imgL = cv2.rectangle(imgL, (731, 2578), (923, 2770), color=(255, 0, 0), thickness=5)

    # cv2.namedWindow('.', cv2.WINDOW_NORMAL)
    # cv2.imshow('.', imgL)
    # cv2.waitKey(0)
    # plt.imshow(imgL)
    # plt.show()

    if res:
        plt.imshow(imgL)
        plt.figure()
        plt.imshow(imgS)
        plt.show()
    elif file:
        p1 = plt.subplot(211)
        p2 = plt.subplot(212)
        p1.imshow(imgL)
        p2.imshow(imgS)
        plt.show()


if __name__ == '__main__':
    timer = Timer()
    opt = get_opt()
    imgL_path = opt.imgL_path
    imgS_path = opt.imgS_path

    with timer:
        res = main_search(imgL_path, imgS_path, opt.search_method)

    print(f'run time: {timer.total_time * 1000:.0f}ms.')
    save_path = f'{opt.result_path}/{Path(imgS_path).stem}_res.json'
    save_res(res, save_path)
    if opt.show == 1:
        show_res(res=res)
