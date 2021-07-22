import os
from pathlib import Path

import cv2
from PIL import Image
import numpy as np


def img_compose_cv2(img, rate, show=False):
    """
        压缩会改变图片尺寸
    """
    h, w, _ = img.shape
    h_new = int(h * rate)
    w_new = int(w * rate)

    img_new = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    return img_new


def test_cv2():
    img_path = '0.jpg'
    img = cv2.imread(img_path)
    img_new = img_compose_cv2(img, 0.5)
    cv2.imshow('1', img)
    cv2.imshow('2', img_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


tmp_path = 'tmp'


def img_compose_pil(img, origin_size, rate=0.8, step=1, quality=80):
    o_size = origin_size
    t_size = int(o_size * rate)
    Path(tmp_path).mkdir(exist_ok=True)
    tmp_img = f'{tmp_path}/{rate}.jpg'
    if o_size < t_size:
        return img
    while o_size > t_size:
        img = Image.open(img_path)
        img.save(tmp_img, quality=quality)
        if quality - step < 0:
            break
        quality -= step
        o_size = os.path.getsize(tmp_img)
    return Image.open(tmp_img)


def crop_img(img_path, box):
    img = Image.open(img_path)

    crop = img.crop(box)
    crop_filename = Path(img_path).name + '_crop' + '.jpg'
    crop_savepath = f'{tmp_path}/{crop_filename}'
    crop.save(crop_savepath)
    crop_size = os.path.getsize(crop_savepath)
    return crop, crop_size

def


if __name__ == '__main__':
    img_path = '0.jpg'
    rates = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    tmp_imgs = []
    img = Image.open(img_path)
    for rate in rates:
        save_path = f'tmp/{rate}.jpg'
        res = img_compose_pil(img, os.path.getsize(img_path), rate=rate)
        tmp_imgs.append(res)
