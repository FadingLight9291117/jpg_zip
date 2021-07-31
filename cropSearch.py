"""
    穷举法搜索一张大图中的小图位置，使用最小MAE判别
"""
import json
import time
from pathlib import Path
import argparse

import cv2
import numpy as np
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

from utils import Timer


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imgL-path', type=str, default='./data/images_2021_07_29/1.jpg', help='大图的文件夹路径')
    parser.add_argument('--imgS-path', type=str, default='./data/images_2021_07_29/1_3.jpg', help='小图的文件夹路径')
    parser.add_argument('--result-path', type=str, default='./result', help='存放结果的文件夹')
    parser.add_argument('--show', action='store', default=True, help='是否显示结果图片')

    return parser.parse_args()


def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))


def search(imgL: np.ndarray, imgS: np.ndarray, show=False) -> dict:
    imgS_h, imgS_w, _ = imgS.shape
    imgL_h, imgL_w, _ = imgL.shape

    minimum = edict({
        'x1': 0,
        'y1': 0,
        'x2': imgS_w,
        'y2': imgS_h,
        # 'mae': mae(crop, img[:imgS_h, :imgS_w]),
        'mae': 1,
    })

    # n = 0
    # crop = img[:imgS_h, :imgS_w]
    t = 30
    for j in range(0, imgL_h - imgS_h + 1, t):

        # cv2.imwrite(f'tmp/{n}.jpg', crop)
        # n += 1

        for i in range(0, imgL_w - imgS_w + 1, t):
            # print(f'i j: ({i}, {j}) target: ({imgL_w - imgS_w}, {imgL_h - imgS_h})')
            crop = imgL[j: j + imgS_h, i: i + imgS_w]
            # crop = img[1430: 1430 + imgS_h, 1258: 1258 + imgS_w]
            this_mae = mae(crop, imgS)
            # print(this_mae)
            if this_mae < minimum.mae:
                minimum.x1 = i
                minimum.y1 = j
                minimum.mae = this_mae
                print(minimum)
                # plt.imshow(crop)
                # plt.show()

    minimum.x2 = minimum.x1 + imgS_w
    minimum.y2 = minimum.y1 + imgS_h
    if show:
        crop = imgL[minimum.y1: minimum.y2, minimum.x1: minimum.x2]
        cv2.imshow('crop', imgS)
        cv2.imshow('crop', crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return dict(minimum)


def transform_img(img: np.array):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float)
    img /= 255
    return img


def main_search(imgL_file, imgS_file, show=False):
    imgL = cv2.imread(str(imgL_file))
    imgS = cv2.imread(str(imgS_file))

    imgL = transform_img(imgL)
    imgS = transform_img(imgS)

    result_ = search(imgL, imgS, show)

    result = {
        'imgL_path': imgL_file,
        'imgS_path': imgS_file,
        'result': result_,
    }
    return result


def save_res(res, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)


def show_res(**kargs):
    res = kargs.get('res')
    file = kargs.get('file')

    if res:
        imgL_path = res['imgL_path']
        imgS_path = res['imgS_path']
        result = res['result']
        result = edict(result)
    elif file:
        with open(file, encoding='utf-8') as f:
            data = json.load(f)
        data = edict(data)
        imgL_path = data.imgL_path
        imgS_path = data.imgS_path
        result = edict(data.result)
    else:
        return

    imgL = cv2.imread(imgL_path)
    imgS = cv2.imread(imgS_path)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    imgL = cv2.rectangle(imgL, (result.x1, result.y1), (result.x2, result.y2), color=(255, 0, 0), thickness=5)

    if res:
        plt.imshow(imgL)
        plt.show()
        plt.imshow(imgS)
        plt.show()
    elif file:
        p1 = plt.subplot(211)
        p2 = plt.subplot(212)
        p1.imshow(imgL)
        p2.imshow(imgS)
        plt.show()


if __name__ == '__main__':
    opt = get_opt()
    imgL_path = opt.imgL_path
    imgS_path = opt.imgS_path
    timer = Timer()
    with timer:
        res = main_search(imgL_path, imgS_path)
    print(f'total time: {timer.average_time():.2F}s.')
    save_path = f'{opt.result_path}/{Path(imgS_path).stem}_res.json'
    save_res(res, save_path)
    if opt.show:
        show_res(res=res)
