import sys
from typing import List

sys.path.append('../')

import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from data.dataset import Dataset
from imageSearch import save_img

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
        img = cv2.resize(img, (int(w // rate), int(h // rate)), cv2.INTER_LINEAR)
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
    return box, conf


if __name__ == '__main__':
    data_dir = '../data/face'
    dataset = Dataset.from_dir(data_dir)
    data = dataset.randOne()
    imgL = data.imgL
    imgS = data.imgS
    gt_box = data.box
    res_box = cv2Search(imgL, imgS)
    save_img(f'1.jpg', imgL, res_box, gt_box)
