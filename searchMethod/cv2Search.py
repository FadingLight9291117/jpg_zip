import sys

sys.path.append('../')
from typing import List
import json
from pathlib import Path

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm

from data.dataset import Dataset
from utils import Timer, save2json

__all__ = ['cv2Search']


def box_mae(imgL, imgS, box):
    s_h, s_w = imgS.shape[:2]
    x1, y1, x2, y2 = box
    b_h = y2 - y1
    b_w = x2 - x1

    if s_h != b_h or s_w != b_w:
        mae = 100000
    else:
        imgR = imgL[y1: y2, x1: x2]
        mae = np.mean(np.abs(imgS - imgR))
    return mae


def show_img(img):
    cv2.imshow('.', img)
    cv2.waitKey(0)


def _trans_img(img, rate):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if rate != 1:
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (int(w // rate), int(h // rate)), cv2.INTER_LINEAR)
        # show_img(img)
    return img


def bound(x, a, b):
    if x < a:
        x = a
    elif x > b:
        x = b
    return x


def _inv_trans(a, rate):
    a = int(a * rate)
    return a


def adaptive_rate(imgS, b_rate, K=50):
    rate = 1
    if b_rate == True:
        h, w = imgS.shape[:2]
        rate = np.min([h / K, w / K])
        rate = np.max([rate, 1])
        # rate = int(rate + 0.5)
        # rate = bound(rate, 3, 4)
        # a = int(np.log2(rate))
        # rate = np.power(2, a + 1)
    return rate


def cv2Search(imgL, imgS, b_rate=True, K=20, mae_threshold=0.05, method=cv2.TM_SQDIFF_NORMED):
    rate = adaptive_rate(imgS, b_rate, K)
    # print(f'rate: {rate}.')
    imgL1 = _trans_img(imgL, rate)
    imgS1 = _trans_img(imgS, rate)
    coeff = cv2.matchTemplate(imgL1, imgS1, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(coeff)
    x1, y1 = min_loc

    s_h, s_w, _ = imgS.shape
    l_h, l_w, _ = imgL.shape
    x1 = _inv_trans(x1, rate)
    y1 = _inv_trans(y1, rate)
    x1 = bound(x1, 0, l_w)
    y1 = bound(y1, 0, l_h)
    x2 = bound(x1 + s_w, 0, l_w)
    y2 = bound(y1 + s_h, 0, l_h)
    box = [x1, y1, x2, y2]

    mae_ = box_mae(imgL, imgS, box)
    conf = 1 - mae_ / mae_threshold
    conf = bound(conf, 0, 1)
    res = {
        'box': box,
        'conf': conf,
    }
    res = edict(res)
    return res


if __name__ == '__main__':
    from imageSearch import save_img, get_metric, box_dist

    data_dir = '../data/face'
    dataset = Dataset.from_dir(data_dir)
    N = 100
    Ks = [i for i in range(10, 100)]
    some_data = dataset[:N]  # 迭代器只能使用一次

    metric_res = []
    Ks_ = tqdm(Ks)
    for K in Ks_:
        timer = Timer()
        dists = []
        for data in some_data:
            with timer:
                res = cv2Search(data.imgL, data.imgS, K=K)
                dist = box_dist(data.box, res.box)
                dists.append(dist)
        metric = get_metric(timer, dists)
        metric['k'] = K
        metric_res.append(metric)

    save_path = Path('') / '..' / 'result' / 'cv2_res' / 'metrics.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save2json(metric_res, save_path)
