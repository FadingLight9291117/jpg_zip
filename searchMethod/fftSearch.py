import sys
from pathlib import Path

sys.path.append('..')

import cv2
import numpy
import numpy as np
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils import Timer
from data.dataset import Dataset

__all__ = ['fftSearch']


def _trans_img(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(float)
    # im = im[:, :, 0]
    return im


def _read_img(img_path):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = _trans_img(im)
    return im


def _save_img(im, save_path):
    im = im.astype(numpy.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, im)


def fftSearch2(imgL, imgS, rate=1):
    imgL_o = imgL
    imgL = _trans_img(imgL)
    imgS = _trans_img(imgS)

    l_h, l_w = imgL.shape[0], imgL.shape[1]
    s_h, s_w = imgS.shape[0], imgS.shape[1]
    # img_tmp = np.zeros_like(imgL)
    # img_tmp[:s_h, :s_w] = imgS
    # imgS = img_tmp

    new_l_h = int(l_h // rate)
    new_l_w = int(l_w // rate)
    new_s_h = int(s_h // rate)
    new_s_w = int(s_w // rate)

    imgL = cv2.resize(imgL, (new_l_w, new_l_h))
    imgS = cv2.resize(imgS, (new_s_w, new_s_h))

    imgS = imgS - np.mean(imgS)

    imgL_f = fft2(imgL)
    imgS_f = fft2(imgS, shape=(new_l_h, new_l_w))

    fc = imgL_f * imgS_f
    # fcn = fc / np.abs(fc)

    pc_matrix = ifft2(fc).real
    # pc_matrix = np.abs(ifft2(fcn))
    max_idx = np.unravel_index(np.argmax(pc_matrix), pc_matrix.shape)

    topT = 3
    pts = np.unravel_index(pc_matrix.flatten().argsort()[::-1][:topT], pc_matrix.shape)
    pts = np.array(pts).T * rate

    boxes = []
    im = imgL_o
    for p in pts:
        x1 = p[1]
        y1 = p[0]
        x2 = x1 + s_w
        y2 = y1 + s_h
        box = [x1, y1, x2, y2]
        boxes.append(box)

    return boxes


def fftSearch(imgL, imgS, rate=1):
    imgL = _trans_img(imgL)
    imgS = _trans_img(imgS)

    l_h, l_w = imgL.shape[0], imgL.shape[1]
    s_h, s_w = imgS.shape[0], imgS.shape[1]
    # img_tmp = np.zeros_like(imgL)
    # img_tmp[:s_h, :s_w] = imgS
    # imgS = img_tmp

    new_l_h = int(l_h // rate)
    new_l_w = int(l_w // rate)
    new_s_h = int(s_h // rate)
    new_s_w = int(s_w // rate)

    imgL = cv2.resize(imgL, (new_l_w, new_l_h))
    imgS = cv2.resize(imgS, (new_s_w, new_s_h))

    # imgL = imgL - np.mean(imgL)
    # imgS = imgS - np.mean(imgS)

    imgL_f = fft2(imgL)
    imgS_f = fft2(imgS, shape=(new_l_h, new_l_w))

    fc = imgL_f * imgS_f.conj()
    fc = fc / (np.abs(fc) + 1e-10)

    pc_matrix = ifft2(fc).real
    # pc_matrix = np.abs(ifft2(fc))
    max_idx = np.unravel_index(np.argmax(pc_matrix), pc_matrix.shape)

    p1 = np.array([max_idx[1], max_idx[0]]) * rate

    box = np.zeros(4)
    box[:2] = p1
    box[2] = p1[0] + s_w
    box[3] = p1[1] + s_h
    box = list(map(int, box))

    return box


def get_mae(img1, img2):
    if img1.shape != img2.shape:
        return -1
    return np.mean(np.abs(img1 - img2))


def fftSearch2_test():
    data_dir = Path('../data/face')
    label_path = data_dir / 'label.json'
    imgL_path = data_dir / 'imgL'
    imgS_path = data_dir / 'imgS'

    save_path = Path('../temp')
    save_path.mkdir(exist_ok=True)

    dataset = Dataset(imgL_path.__str__(), imgS_path.__str__(), label_path.__str__())

    N = 100
    rate = 2

    datas = dataset.randN(N)
    datas = tqdm(datas, total=N)

    for data in datas:
        imgL = data.imgL
        imgS = data.imgS
        box = data.box
        res_boxes = fftSearch2(imgL, imgS, rate)

        im = imgL
        im = cv2.rectangle(im, box[:2], box[2:], color=(255, 0, 0), thickness=2)
        for i, res_box in enumerate(res_boxes):
            rand_color = np.random.randint(255, size=(3,)).astype(int)
            rand_color = tuple(map(int, rand_color))

            im1 = np.array(Image.fromarray(imgL).crop(box), dtype=float) / 255
            im2 = np.array(Image.fromarray(imgL).crop(res_box), dtype=float) / 255
            mae = get_mae(im1, im2)

            im = cv2.rectangle(im, res_box[:2], res_box[2:], color=rand_color, thickness=2)
            im = cv2.putText(im, f'{i} - {mae:.2f}', res_box[:2], cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 0, 0))
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        save_img_path = save_path / f'{rate}' / data.imgS_name
        save_img_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(save_img_path.__str__(), im)


def fftSearch2_test2():
    rate = 1
    data_dir = Path('../data/face')
    label_path = data_dir / 'label.json'
    imgL_path = data_dir / 'imgL'
    imgS_path = data_dir / 'imgS'

    save_path = Path('../temp')
    save_path.mkdir(exist_ok=True)

    dataset = Dataset(imgL_path.__str__(), imgS_path.__str__(), label_path.__str__())

    imgS_name = '63_crop_0.jpg'

    data = None
    for data_ in dataset:
        if data_.imgS_name == imgS_name:
            data = data_

    h, w, _ = data.imgL.shape

    imgL = data.imgL[h // 2:, : w // 2]
    imgS = data.imgS
    box = data.box
    res_boxes = fftSearch(imgL, imgS, rate)

    im = imgL
    # im = cv2.rectangle(im, box[:2], box[2:], color=(255, 0, 0), thickness=2)
    for i, res_box in enumerate(res_boxes[:1]):
        rand_color = np.random.randint(255, size=(3,)).astype(int)
        rand_color = tuple(map(int, rand_color))

        im1 = np.array(Image.fromarray(imgL).crop(box), dtype=float) / 255
        im2 = np.array(Image.fromarray(imgL).crop(res_box), dtype=float) / 255
        mae = get_mae(im1, im2)

        im = cv2.rectangle(im, res_box[:2], res_box[2:], color=rand_color, thickness=2)
        im = cv2.putText(im, f'{i} - {mae:.2f}', res_box[:2], cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 0, 0))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    save_img_path = save_path / f'{rate}' / data.imgS_name
    save_img_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(save_img_path.__str__(), im)
    cv2.imwrite(str(save_img_path.parent / 'imgS.jpg'), imgS)
    cv2.imwrite(str(save_img_path.parent / 'imgL.jpg'), imgL)


if __name__ == '__main__':
    fftSearch2_test()
