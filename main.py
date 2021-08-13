import functools
import os
import random
import time
from pathlib import Path
import json
import argparse
from typing import List

import cv2
from PIL import Image
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import pretty_print, number_formatter, Number
from data.dataset import Dataset


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str,
                        default='data/face', help='原始图片文件夹路径。')
    parser.add_argument('--chart-path', type=str,
                        default='charts', help='存放保存的图表')
    # parser.add_argument('--result-path', type=str, default='result/')

    return parser.parse_args()


tmp_number = Number()
tmp_path = 'tmp'
Path(tmp_path).mkdir(exist_ok=True)


def img_compose_cv2(img, quality, show=False):
    """
        opencv
    """
    save_path = Path(tmp_path) / f'{tmp_number()}_{quality}.jpg'
    cv2.imwrite(str(save_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    img_res = cv2.imread(str(save_path))

    if show:
        img.imshow(save_path.name, img_res)

    return img_res


def trans_img(img):
    img = img.astype(np.float32)
    img = img / 255
    return img


def crop_img(img, bbox, save=False):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if save:
        crop_filename = f'{tmp_number()}_crop.jpg'
        crop_save_path = f'{tmp_path}/{crop_filename}'
        cv2.imwrite(crop_save_path, crop)
    return crop


def get_metrics(img, o_img):
    img = trans_img(img)
    o_img = trans_img(o_img)
    img_sub = img - o_img
    mae = np.mean(np.abs(img_sub))
    mae = float(mae)
    metrics = {
        'mae': mae,
    }
    return edict(metrics)


def get_imgs(paths):
    img_paths = Path(paths).glob('*.jpg')
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        imgs.append(img)

    return list(map(transform_img, imgs))


def main():
    epochs = 100
    qualities = [i * 10 for i in range(1, 11)]
    opt = get_opt()
    data_dir = opt.data_dir

    save_path = Path('result') / 'main' / Path(data_dir).name
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_dir(data_dir)
    qualities_ = tqdm(qualities)
    maes = []
    for quality in qualities_:
        maes_ = []
        for _ in range(epochs):
            rand_idx = np.random.randint(len(dataset))
            data = dataset[rand_idx]
            imgS = data.imgS
            imgS_compressed = img_compose_cv2(imgS, quality)

            mae = get_metrics(imgS, imgS_compressed).mae
            maes_.append(mae)
        maes.append(maes_)
    res = {
        'qualities': qualities,
        'maes': maes,
    }
    mae_save_path = save_path / 'mae.json'
    with mae_save_path.open('w', encoding='utf-8') as f:
        json.dump(res, f)
    print(f'save in {str(mae_save_path)}.')
    return res


def error_main():
    epochs = 100
    qualities = [i * 10 for i in range(1, 11)]
    opt = get_opt()
    data_dir = Path(opt.data_dir)

    save_path = Path('result') / 'error_main' / Path(data_dir).name
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_dir(data_dir.__str__())

    maes = []
    epochs = 100
    qualities_ = tqdm(qualities)
    for quality in qualities_:
        maes_ = []
        for i in range(epochs):
            rand_idx = np.random.randint(len(dataset))
            data = dataset[rand_idx]
            imgS = data.imgS
            crop = rand_crop(data.imgL, imgS.shape[:2])
            imgS_compressed = img_compose_cv2(imgS, quality)

            mae = get_metrics(crop, imgS_compressed).mae
            maes_.append(mae)
        maes.append(maes_)
    res = {
        'qualities': qualities,
        'maes': maes,
    }
    mae_save_path = save_path / 'mae.json'
    with mae_save_path.open('w', encoding='utf-8') as f:
        json.dump(res, f)
    print(f'save in {str(mae_save_path)}.')
    return res


def rand_crop(img: np.ndarray, shape):
    """
    随机中心点，不同尺寸的吧
    """
    img_h, img_w, _ = img.shape
    c_h, c_w = shape

    point1 = (np.random.randint(img_w - c_w),
              np.random.randint(img_h - c_h))
    point2 = (point1[0] + c_w, point1[1] + c_h)
    box = [*point1, *point2]
    return img[box[3] - box[1], box[2] - box[0]]


def get_bbox_name(bbox):
    return f'{bbox[2] - bbox[0]}X{bbox[3] - bbox[1]}'


def transform_img(img: np.array):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    # img /= 255
    return img


if __name__ == '__main__':
    # main()
    error_main()
