from typing import Dict, List

import cv2
import numpy as np
from easydict import EasyDict as edict

__all__ = ['spaceSearch']


def _mae(img1, img2):
    return np.mean(np.abs(img1 - img2))


def _refined_search(imgL: np.ndarray, imgS: np.ndarray, init_box, stride: int) -> Dict[str, int]:
    assert imgS.shape[0] == init_box[3] - init_box[1]
    assert imgS.shape[1] == init_box[2] - init_box[0]

    crop_w = init_box[2] - init_box[0]
    crop_h = init_box[3] - init_box[1]

    init_crop = {
        'x1': init_box[0] - stride if init_box[0] - stride > 0 else 0,
        'y1': init_box[1] - stride if init_box[1] - stride > 0 else 0,
    }
    init_crop['x2'] = init_crop['x1'] + crop_w
    init_crop['y2'] = init_crop['y1'] + crop_h
    init_crop = edict(init_crop)

    res = {
        'x1': init_crop.x1,
        'y1': init_crop.y1,
        'x2': init_crop.x2,
        'y2': init_crop.y2,
        'mae': 1,
    }
    res = edict(res)
    res.mae = _mae(imgL[res.y1: res.y2, res.x1: res.x2], imgS)

    for i in range(stride * 2):  # 行
        for j in range(stride * 2):  # 列
            box = [
                init_crop.x1 + j,
                init_crop.y1 + i,
                init_crop.x2 + j,
                init_crop.y2 + i,
            ]
            if box[2] > imgL.shape[1] or box[3] > imgL.shape[0]:
                break
            crop = imgL[box[1]:  box[3], box[0]: box[2]]
            this_mae = _mae(crop, imgS)
            if this_mae < res.mae:
                res.x1, res.y1, res.x2, res.y2 = box
                res.mae = this_mae
                # print(res)
    return res


def spaceSearch(imgL: np.ndarray, imgS: np.ndarray, stride=40, refined=True) -> List[int]:
    imgL = _transform_img(imgL)
    imgS = _transform_img(imgS)

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
    # t = imgL_h * imgL_w / (imgS_w * imgS_h)
    t = stride
    for j in range(0, imgL_h - imgS_h + 1, t):

        # cv2.imwrite(f'tmp/{n}.jpg', crop)
        # n += 1

        for i in range(0, imgL_w - imgS_w + 1, t):
            # print(f'i j: ({i}, {j}) target: ({imgL_w - imgS_w}, {imgL_h - imgS_h})')
            crop = imgL[j: j + imgS_h, i: i + imgS_w]
            # crop = img[1430: 1430 + imgS_h, 1258: 1258 + imgS_w]
            this_mae = _mae(crop, imgS)
            # print(this_mae)
            if this_mae < minimum.mae:
                minimum.x1 = i
                minimum.y1 = j
                minimum.x2 = i + imgS_w
                minimum.y2 = j + imgS_h
                minimum.mae = this_mae
                # print(minimum)
                # plt.imshow(crop)
                # plt.show()
    if refined:
        min_box = [
            minimum.x1,
            minimum.y1,
            minimum.x2,
            minimum.y2,
        ]
        minimum = _refined_search(imgL, imgS, min_box, t)
    return [minimum.x1, minimum.y1, minimum.x2, minimum.y2]


def _transform_img(img: np.array):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img /= 255
    return img
