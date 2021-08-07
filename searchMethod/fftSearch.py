import cv2
import numpy
import numpy as np
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt

from utils import Timer

__all__ = ['fftSearch']


def _trans_img(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(float)
    # im = im[:, :, 0]
    return im


def _read_img(img_path):
    im = cv2.imread(img_path)
    im = _trans_img(im)
    return im


def _save_img(im, save_path):
    im = im.astype(numpy.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, im)


def fftSearch(imgL, imgS, rate=5):
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

    imgL_f = fft2(imgL)
    imgS_f = fft2(imgS, shape=(new_l_h, new_l_w))

    fc = imgL_f * imgS_f.conj()
    fcn = fc / np.abs(fc)

    pc_matrix = ifft2(fcn).real
    max_idx = np.unravel_index(np.argmax(pc_matrix), pc_matrix.shape)

    p1 = (max_idx[1], max_idx[0])
    p2 = (max_idx[1] + imgS.shape[1], max_idx[0] + imgS.shape[0])

    box = [*p1, *p2]
    box = [int(i * rate) for i in box]  # 这里box坐标放大后会跟imgS的size对不上，有几个pixel的差异
    box[2] = box[0] + s_w
    box[3] = box[1] + s_h
    return box


if __name__ == '__main__':
    timer = Timer()
    imgL_path = 'data/images_2021_07_29/1.jpg'
    imgS_path = 'data/images_2021_07_29/1_1.jpg'
    imgL = _read_img(imgL_path)
    imgS = _read_img(imgS_path)
    with timer:
        res = fftSearch(imgL, imgS)
    print(f'total time is {timer.average_time() * 1000:.0f}ms.')
    res = list(res.values())
    im = cv2.rectangle(imgL, res[:2], res[2:4], color=(255, 0, 0), thickness=10)
    plt.imshow(im)
    plt.show()