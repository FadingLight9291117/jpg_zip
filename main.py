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

from utils import pretty_print, number_formatter, Number


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-path', type=str, default='data/imgs', help='原始图片文件夹路径。')
    parser.add_argument('--chart-path', type=str, default='charts', help='存放保存的图表')
    # parser.add_argument('--result-path', type=str, default='result/')

    return parser.parse_args()


tmp_number = Number()
tmp_path = 'tmp'
Path(tmp_path).mkdir(exist_ok=True)


def img_compose_cv2(img, rate, show=False):
    """
        opencv
    """
    save_quality = int(100 * rate)
    save_path = Path(tmp_path) / f'{tmp_number()}_{rate}.jpg'
    cv2.imwrite(str(save_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), save_quality])

    img_res = cv2.imread(str(save_path))

    if show:
        img.imshow(save_path.name, img_res)

    return img_res


def crop_img(img, bbox, save=False):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if save:
        crop_filename = f'{tmp_number()}_crop.jpg'
        crop_save_path = f'{tmp_path}/{crop_filename}'
        cv2.imwrite(crop_save_path, crop)
    return crop


def get_metrics(img, o_img):
    img_sub = img - o_img
    mae = np.mean(np.abs(img_sub))

    metrics = {
        'mae': mae / 255,
    }
    return edict(metrics)


def get_imgs(paths):
    img_paths = Path(paths).glob('*.jpg')
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        imgs.append(img)

    return list(map(transform_img, imgs))


def plot_img(df: pd.DataFrame, save_path, show=False):
    # plt.title(r'不同尺寸不同压缩率下的正确图片MAE')
    plt.xlabel('compose rate')
    plt.ylabel('MAE')
    plt.ylim(0, 1)
    for crop in df.index:
        data = df.loc[crop]
        plt.plot(df.columns, data, label=crop)
    plt.legend()  # 图例展示位置，数字代表第几象限
    plt.savefig(f'{save_path}/{time.time() / 100:.2f}.jpg')
    if show:
        plt.show()
    plt.clf()


rates = [i / 10 for i in range(1, 11)]
anchors = [
    [100, 100],
    [200, 100],
    [100, 200],
    [150, 300],
    [300, 150],
    [300, 450],
]


def main():
    opts = get_opt()
    imgs_path = opts.img_path
    charts_path = opts.chart_path
    imgLs = get_imgs(imgs_path)

    df: pd.DataFrame
    df_data = np.zeros((len(imgLs), len(rates)))
    for i, imgL in enumerate(imgLs):
        for j, rate in enumerate(rates):
            # 对于每个压缩率产生n张不同的位置的大小的crop
            boxes = rand_crops(imgL)
            imgSs = [crop_img(imgL, box) for box in boxes]

            crops_composed = [img_compose_cv2(imgS, rate) for imgS in imgSs]
            maes_ = [get_metrics(img, o_img).mae for img, o_img in zip(crops_composed, imgSs)]
            mae = np.mean(maes_)
            df_data[i, j] = mae

    df_idx = [f'image {i}' for i in range(len(imgLs))]
    df = pd.DataFrame(df_data, index=df_idx, columns=rates)

    plot_img(df, charts_path, show=True)


def error_main():
    opts = get_opt()
    imgs_path = opts.img_path
    charts_path = opts.chart_path
    imgLs = get_imgs(imgs_path)

    df: pd.DataFrame
    df_data = np.zeros((len(imgLs), len(rates)))
    for i, imgL in enumerate(imgLs):
        for j, rate in enumerate(rates):
            # 对于每个压缩率产生n张不同的位置的大小的crop
            boxes = rand_crops(imgL)
            imgSs = [crop_img(imgL, box) for box in boxes]

            error_boxes = rand_crops(imgL)
            error_imgSs = [crop_img(imgL, box) for box in error_boxes]

            crops_composed = [img_compose_cv2(imgS, rate) for imgS in imgSs]
            maes_ = [get_metrics(img, o_img).mae for img, o_img in zip(crops_composed, error_imgSs)]
            mae = np.mean(maes_)
            df_data[i, j] = mae

    df_idx = [f'image {i}' for i in range(len(imgLs))]
    df = pd.DataFrame(df_data, index=df_idx, columns=rates)

    plot_img(df, charts_path, show=True)


def rand_crops(img: np.ndarray):
    """
    随机中心点，不同尺寸的吧
    """
    img_h, img_w, _ = img.shape
    boxes = []
    for anchor in anchors:
        while True:
            w, h = anchor
            x, y = np.random.rand(2, )
            x = int(x * img_w)
            y = int(y * img_h)
            x1, x2 = x - w // 2, x + w // 2
            y1, y2 = y - h // 2, y + h // 2
            if x1 >= 0 and x2 <= img_w and y1 >= 0 and y2 <= img_h:
                break
        boxes.append((x1, y1, x2, y2))

    return boxes


def get_bbox_name(bbox):
    return f'{bbox[2] - bbox[0]}X{bbox[3] - bbox[1]}'


def transform_img(img: np.array):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    # img /= 255
    return img


if __name__ == '__main__':
    main()
    error_main()
