import os
import time
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
    img_new = cv2.resize(img_new, (w, h), interpolation=cv2.INTER_AREA)
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
Path(tmp_path).mkdir(exist_ok=True)


def img_compose_pil(img, origin_size, rate=0.8, step=1, quality=80):
    img = Image.fromarray(img)
    o_size = origin_size
    t_size = int(o_size * rate)
    Path(tmp_path).mkdir(exist_ok=True)
    tmp_img = f'{tmp_path}/{rate}.jpg'
    img.save(tmp_img)
    if o_size < t_size:
        return img
    while o_size > t_size:
        img = Image.open(tmp_img)
        img.save(tmp_img, quality=quality)
        if quality - step < 0:
            break
        quality -= step
        o_size = os.path.getsize(tmp_img)
    img_ = Image.open(tmp_img)
    img_ = np.array(img_)
    return img_


def crop_img(img_path, bbox):
    img = Image.open(img_path)

    crop = img.crop(bbox)
    crop_filename = Path(img_path).stem + f'{time.time()}_crop.jpg'
    crop_savepath = f'{tmp_path}/{crop_filename}'
    crop.save(crop_savepath)
    crop_size = os.path.getsize(crop_savepath)
    crop = np.array(crop)
    return crop, crop_size


def cover_img(img_path, crop, bbox, save=False):
    img = Image.open(img_path)
    img_np = np.array(img)
    crop_np = np.array(crop)
    img_np[bbox[1]:bbox[3], bbox[0]:bbox[2]] = crop_np
    if save:
        save_path = f'{tmp_path}/new_{time.time()}.jpg'
        save_img(img_np, save_path)
    return img_np


def get_metrics(imgs, o_img):
    imgs_dis = imgs - o_img
    maes = np.zeros(imgs.shape[0])
    for i in range(len(imgs)):
        maes[i] = np.mean(np.abs(imgs_dis[i]))
    return maes


def main():
    img_path = '0.jpg'
    rates = [0.9, 0.7, 0.5, 0.3, 0.1]
    bboxes = [
        (0, 0, 100, 100),
        (0, 0, 300, 300),
        (0, 0, 500, 500),
        (0, 0, 600, 600),
    ]
    for bbox in bboxes:
        new_imgs = []

        crop, crop_size = crop_img(img_path, bbox)
        for rate in rates:
            # crop_new = img_compose_pil(crop, crop_size, rate=rate)
            crop_new = img_compose_cv2(crop, rate=rate)
            new_img = cover_img(img_path, crop_new, bbox, save=True)
            new_imgs.append(new_img)

        img = Image.open(img_path)
        img_np = np.array(img)
        new_imgs_np = np.array([np.array(img) for img in new_imgs])

        maes = get_metrics(new_imgs_np, img_np)

        maes_dict = {}
        for i, rate in enumerate(rates):
            maes_dict[rate] = maes[i]
        print(f'origin image size: {img.size}')
        print(f'bbox: {bbox}')
        print(maes_dict)


def save_img(img, save_path):
    img = Image.fromarray(img)
    img.save(save_path)


if __name__ == '__main__':
    # test_cv2()
    main()
