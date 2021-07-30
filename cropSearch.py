"""
    穷举法搜索一张大图中的小图位置，使用最小MAE判别
"""
import time
from pathlib import Path
import argparse

import cv2
import numpy as np
from easydict import EasyDict as edict


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-path', type=str, default='./data/images_2021_07_29/1.jpg', help='大图的文件夹路径')
    parser.add_argument('--crop-path', type=str, default='./data/images_2021_07_29/1_1.jpg', help='小图的文件夹路径')

    return parser.parse_args()


def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))


def search(img: np.ndarray, crop: np.ndarray, show=False) -> dict:
    crop_h, crop_w, _ = crop.shape
    img_h, img_w, _ = img.shape

    minimum = edict({
        'x1': 0,
        'y1': 0,
        'x2': crop_w,
        'y2': crop_h,
        'mae': mae(crop, img[:crop_h, :crop_w]),
    })
    for i in range(img_w - crop_w):
        for j in range(img_h - crop_h):
            print(f'i j: ({i}, {j}) target: ({img_w - crop_w}, {img_h - crop_h})')
            img_crop = img[j: j + crop_h, i: i + crop_w]
            this_mae = mae(img_crop, crop)
            if this_mae < minimum.mae:
                minimum.x1 = i
                minimum.y1 = j
                minimum.mae = this_mae
    minimum.x2 = minimum.x1 + crop_w
    minimum.y2 = minimum.y1 + crop_h
    if show:
        img_crop = img[minimum.y1: minimum.y2, minimum.x1: minimum.x2]
        cv2.imshow('crop', crop)
        cv2.imshow('img_crop', img_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return dict(minimum)


def main_search(image_file, crop_file, show=False):
    image_file = Path(image_file)
    crop_file = Path(crop_file)

    img_name = image_file.name
    crop_name = crop_file.name

    img = cv2.imread(image_file.__str__())
    crop = cv2.imread(crop_file.__str__())

    result = search(img, crop, show)

    return {
        'img_name': img_name,
        'crop_name': crop_name,
        'result': result,
    }


if __name__ == '__main__':
    opt = get_opt()
    image_path = opt.image_path
    crop_path = opt.crop_path
    begin = time.time()
    try:
        res = main_search(image_path, crop_path, show=True)
    except Exception:
        print(Exception.__str__())
    finally:
        end = time.time()
        print(f'totalt time is {end - begin:.2f}')
