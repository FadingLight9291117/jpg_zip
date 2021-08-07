"""
    穷举法搜索一张大图中的小图位置，使用最小MAE判别
"""
import json
import time
from pathlib import Path
import argparse
from typing import List, Dict

import cv2
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm

from utils import Timer
from searchMethod import fftSearch, spaceSearch, fftpSearch
from data.dataset import Dataset


class SearchMethodNotFoundException(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'{self.name} not found.'


def main_search(imgL_file, imgS_file, search_method):
    imgL = cv2.imread(str(imgL_file))
    imgS = cv2.imread(str(imgS_file))

    if search_method == 'space':
        search_method = spaceSearch.spaceSearch
    elif search_method == 'fft':
        search_method = fftSearch.fftSearch
    elif search_method == 'fftp':
        search_method = fftpSearch.fftpSearch
    else:
        raise SearchMethodNotFoundException(search_method)

    box = search_method(imgL, imgS)

    result = {
        'imgL_file': imgL_file,
        'imgS_file': imgS_file,
        'box': box,
    }
    return result


def save_res(res, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)
        # print(f'result save in "{save_path}"')


def show_res(imgL, res_box, gt_box, save=False):
    im = cv2.cvtColor(imgL, cv2.COLOR_RGB2BGR)
    im = cv2.rectangle(im, res_box[:2], res_box[2:], color=(0, 0, 255), thickness=2)
    im = cv2.rectangle(im, gt_box[:2], gt_box[2:], color=(0, 255, 0), thickness=2)
    cv2.imshow('show', im)
    cv2.waitKey(0)


def metric(res_box, gt_box):
    dist = np.sqrt(np.power(res_box[0] - gt_box[0], 2) + np.power(res_box[1] - gt_box[1], 2))
    return dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--imgL-dir', type=str, default='./data/celefaces/imgL', help='大图的文件夹路径')
    parser.add_argument('--imgS-dir', type=str, default='./data/celefaces/imgS', help='小图的文件夹路径')
    parser.add_argument('--result-path', type=str, default='./result', help='存放结果的文件夹')
    parser.add_argument('--label-path', type=str, default='./data/celefaces/label.json')
    parser.add_argument('--search-method', type=str, default='fftp', help='搜索方法')
    parser.add_argument('--show', action='store_true', default=False, help='是否显示结果图片')
    parser.add_argument('--enable-log', action='store_true', default=False, help='是否显示log')

    opt = parser.parse_args()

    imgL_dir = opt.imgL_dir
    imgS_dir = opt.imgS_dir
    label_path = opt.label_path
    search_method = opt.search_method

    dataset = Dataset(imgL_dir, imgS_dir, label_path)

    timer = Timer()

    N = len(dataset)
    datas = tqdm(dataset.randN(N), total=N)
    dists = []

    for data in datas:
        imgL_path = f'{imgL_dir}/{data.imgL_name}'
        imgS_path = f'{imgS_dir}/{data.imgS_name}'
        save_path = f'{opt.result_path}/{Path(imgS_path).stem}_res.json'

        with timer:
            res = main_search(imgL_path, imgS_path, search_method)
        res = edict(res)
        save_res(res, save_path)
        dist = metric(res.box, data.box)
        dists.append(dist)
        if opt.enable_log:
            print(f'run time: {timer.this_time() * 1000:.0f}ms.')
            print(f'dist: {dist:.2f}')
        if opt.show:
            show_res(data.imgL, res_box=res.box, gt_box=data.box)

    print(f'average dist: {np.mean(dists)}.')
    print(f'average time: {timer.average_time() * 100:.0f}ms.')
