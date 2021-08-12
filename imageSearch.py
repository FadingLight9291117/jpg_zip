"""
    穷举法搜索一张大图中的小图位置，使用最小MAE判别
"""
import json
import time
from pathlib import Path
import argparse
from typing import List, Dict
import functools

import cv2
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from utils import Timer, pretty_print
from searchMethod import fftSearch, spaceSearch, fftpSearch, cv2Search
from data.dataset import Dataset


def _trans_img(img):
    img = img.astype(np.float32)
    img = img / 255
    return img


class SearchMethodNotFoundException(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'{self.name} not found.'


def main_search(imgL_file, imgS_file, search_method, rate=1):
    imgL = cv2.imread(str(imgL_file))
    imgS = cv2.imread(str(imgS_file))
    imgL = _trans_img(imgL)
    imgS = _trans_img(imgS)

    if search_method == 'space':
        search_method = functools.partial(spaceSearch.spaceSearch, stride=rate)
    elif search_method == 'fft':
        search_method = functools.partial(fftSearch.fftSearch, rate=rate)
    elif search_method == 'fftp':
        search_method = functools.partial(fftpSearch.fftpSearch, rate=rate)
    elif search_method == 'cv2':
        search_method = cv2Search.cv2Search
    else:
        raise SearchMethodNotFoundException

    box, rate = search_method(imgL, imgS)

    result = {
        'imgL_file': imgL_file,
        'imgS_file': imgS_file,
        'box': box,
        'param': rate,
    }
    return result


def save_res(res, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)
        # print(f'result save in "{save_path}"')


def save_img(img_path, imgL, res_box, gt_box):
    im = cv2.cvtColor(imgL, cv2.COLOR_RGB2BGR)
    im = cv2.rectangle(im, res_box[:2], res_box[2:], color=(0, 0, 255), thickness=2)
    im = cv2.rectangle(im, gt_box[:2], gt_box[2:], color=(0, 255, 0), thickness=2)
    cv2.imwrite(img_path, im)


def show_res(imgL, res_box, gt_box):
    im = cv2.cvtColor(imgL, cv2.COLOR_RGB2BGR)
    im = cv2.rectangle(im, res_box[:2], res_box[2:], color=(0, 0, 255), thickness=2)
    im = cv2.rectangle(im, gt_box[:2], gt_box[2:], color=(0, 255, 0), thickness=2)
    cv2.imshow('show', im)
    cv2.waitKey(0)


def metric(res_box, gt_box):
    dist = np.sqrt(np.power(res_box[0] - gt_box[0], 2) + np.power(res_box[1] - gt_box[1], 2))
    return dist


def get_metric(timer, dists, t):
    metrics = {
        'acc(20)': np.mean(dists < t),
        'average dist(pixel)': f'{np.mean(dists):.2f}',
        'average time(ms)': f'{timer.average_time() * 1000: .2f}'
    }
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str, default='./data/face',
                        help='数据的路径，必须有imgL文件夹，imgS文件夹，和label.json标签文件')
    parser.add_argument('--result-path', type=str, default='./result', help='存放结果的文件夹')
    parser.add_argument('--search-method', type=str, default='cv2', help='搜索方法')
    parser.add_argument('--rate', type=int, default=20, help='搜索方法的参数')
    parser.add_argument('--save', action='store_true', default=True, help='保存结果图片')
    parser.add_argument('--show', action='store_true', default=False, help='是否显示结果图片')
    parser.add_argument('--enable-log', action='store_true', default=False, help='是否显示log')

    opt = parser.parse_args()

    dataset_dir = opt.dataset_dir
    dataset_name = Path(dataset_dir).name
    imgL_dir = str(Path(dataset_dir) / 'imgL')
    imgS_dir = str(Path(dataset_dir) / 'imgS')
    label_path = str(Path(dataset_dir) / 'label.json')
    search_method = opt.search_method
    rate = opt.rate

    dataset = Dataset(imgL_dir, imgS_dir, label_path)

    timer = Timer()

    N = 500
    datas = dataset[:N]
    datas = tqdm(datas, total=N)
    dists = []

    for data in datas:
        imgL_path = f'{imgL_dir}/{data.imgL_name}'
        imgS_path = f'{imgS_dir}/{data.imgS_name}'
        save_path = f'{opt.result_path}/{Path(imgS_path).stem}_res.json'

        with timer:
            res = main_search(imgL_path, imgS_path, search_method, rate)

        res = edict(res)
        # save_res(res, save_path)
        dist = metric(res.box, data.box)
        dists.append(dist)
        if opt.enable_log:
            print(f'run time: {timer.this_time() * 1000:.0f}ms.')
            print(f'dist: {dist:.2f}')
        if opt.show:
            show_res(data.imgL, res_box=res.box, gt_box=data.box)
        if opt.save:
            save_img_path = Path('result') / dataset_name / search_method / data.imgS_name
            save_img_path.parent.mkdir(exist_ok=True, parents=True)
            save_img(str(save_img_path), data.imgL, res_box=res.box, gt_box=data.box)
        # print(res, dist)

    dists = np.array(dists)
    t = opt.rate

    metrics = get_metric(timer, dists, t)
    pretty_print(metrics)
