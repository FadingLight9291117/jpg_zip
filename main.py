import functools
import os
import random
import time
from pathlib import Path
import json
import argparse

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
    parser.add_argument('--tmp-path', type=str, default='tmp', help='存放临时图片的文件夹。')

    return parser.parse_args()


opt = get_opt()

tmp_path = opt.tmp_path
Path(tmp_path).mkdir(exist_ok=True)

tmp_number = Number()
imgs_path = opt.img_path


def img_compose_cv2_(img, rate, show=False):
    """
        压缩会改变图片尺寸
    """

    h, w, _ = img.shape
    h_new = int(h * rate)
    w_new = int(w * rate)

    img_new = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    img_new = cv2.resize(img_new, (w, h), interpolation=cv2.INTER_AREA)

    return img_new


def img_compose_cv2(img, rate, show=False):
    """
        opencv
    """
    save_quality = int(100 * rate)
    save_path = Path('tmp') / f'{tmp_number()}_{rate}.jpg'
    cv2.imwrite(str(save_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), save_quality])

    img_res = cv2.imread(str(save_path))

    if show:
        img.imshow(save_path.name, img_res)

    return img_res


def crop_img(img, bbox, save=False):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2, :]
    crop_filename = f'{tmp_number()}_crop.jpg'
    crop_save_path = f'{tmp_path}/{crop_filename}'
    if save:
        cv2.imwrite(crop_save_path, crop)
    return crop


def cover_img(img, crop, bbox, save=False):
    img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = crop
    if save:
        save_path = f'{tmp_path}/new_{tmp_number()}.jpg'
        save_img(img, save_path)
    return img


def get_metrics(img, o_img):
    img_sub = img - o_img
    mae = np.mean(np.abs(img_sub))

    metrics = {
        'mae': mae / 255
    }
    return edict(metrics)


def get_imgs(paths):
    img_paths = Path(paths).glob('*')
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        imgs.append(img)
    return imgs


def plot_img(df: pd.DataFrame):
    # plt.title(r'不同尺寸不同压缩率下的正确图片MAE')
    plt.xlabel('compose rate')
    plt.ylabel('MAE')
    plt.ylim(0, 1)
    for crop in df.index:
        data = df.loc[crop]
        plt.plot(df.columns, data, label=crop)
    plt.legend()  # 图例展示位置，数字代表第几象限
    plt.savefig(f'{time.time() :.2f}.jpg')
    plt.show()


rates = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
bboxes = [
    (0, 0, 100, 100),
    (0, 0, 120, 240),
    (0, 0, 250, 280),
    (0, 0, 300, 300),
    (0, 0, 350, 350),
]


def error_test():
    imgs = get_imgs(imgs_path)

    dfs = []
    for img in imgs:
        crops = []
        crops_composed = []
        error_crops = []
        for bbox in bboxes:
            crop = crop_img(img, bbox)
            crops.append(crop)

            error_bbox = np.array(bbox) + random.randint(1, 100)
            error_crop = crop_img(img, error_bbox)
            error_crops.append(error_crop)

            crop_news = []
            for rate in rates:
                crop_new = img_compose_cv2(crop, rate=rate)
                crop_news.append(crop_new)
            crops_composed.append(crop_news)

        maes = np.zeros((len(bboxes), len(rates)))
        for i, (crop, crop_composed_list) in enumerate(zip(error_crops, crops_composed)):
            for j, crop_new in enumerate(crop_composed_list):
                metric = get_metrics(crop, crop_new)
                mae = metric.mae
                maes[i, j] = mae
        df = pd.DataFrame(maes, index=[trans_bbox(box) for box in bboxes], columns=rates)

        dfs.append(df)
    df_ = sum(dfs) / len(dfs)
    plot_img(df_)


def main():
    imgs = get_imgs(imgs_path)
    dfs = []
    for img in imgs:
        crops = []
        crops_composed = []
        for bbox in bboxes:
            crop = crop_img(img, bbox)
            crops.append(crop)
            crop_news = []
            for rate in rates:
                crop_new = img_compose_cv2(crop, rate=rate)
                crop_news.append(crop_new)
            crops_composed.append(crop_news)

        maes = np.zeros((len(bboxes), len(rates)))
        for i, (crop, crop_composed_list) in enumerate(zip(crops, crops_composed)):
            for j, crop_new in enumerate(crop_composed_list):
                metric = get_metrics(crop, crop_new)
                mae = metric.mae
                maes[i, j] = mae
        df = pd.DataFrame(maes, index=[trans_bbox(box) for box in bboxes], columns=rates)

        dfs.append(df)
    df_ = sum(dfs) / len(dfs)
    plot_img(df_)

    # print('======================not origin img=======================')
    #
    # crops = []
    # for bbox in bboxes:
    #     crop = crop_img(img_paths[0], bbox)
    #     for rate in rates:
    #         crop_new = img_compose_cv2(crop, rate=rate)
    #         crops.append(crop_new)
    # crops_np = np.array([np.array(img) for img in crops])
    #
    # for img_path in img_paths[1:]:
    #     metrics = []
    #     for bbox in bboxes:
    #         other_crop = crop_img(img_path, bbox)
    #
    #         maes = get_metrics(crops_np, other_crop).tolist()
    #         metrics.append(maes)
    #
    #     df = pd.DataFrame(metrics, index=list(map(trans_bbox, bboxes)), columns=rates)
    #     print(df)


def trans_bbox(bbox):
    return f'{bbox[2]}X{bbox[3]}'


def save_img(img, save_path):
    img = Image.fromarray(img)
    img.save(save_path)


if __name__ == '__main__':
    # test_cv2()
    main()
    error_test()
