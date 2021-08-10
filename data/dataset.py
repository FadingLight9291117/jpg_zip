import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from easydict import EasyDict as edict
from typing import List
import cv2

__all__ = ['Dataset']


class Dataset:
    @dataclass
    class _Data:
        imgL_name: str
        imgS_name: str
        imgL: np.ndarray
        imgS: np.ndarray
        box: np.ndarray

    def __init__(self, imgL_dir, imgS_dir, label_path):
        self.imgL_path = imgL_dir
        self.imgS_path = imgS_dir
        with open(label_path, encoding='utf-8') as f:
            self.labels = json.load(f)

    def _trans_img(self, img: np.ndarray):
        # img = img.astype(np.float)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def randOne(self):
        rand_idx = np.random.randint(len(self))
        data = self[rand_idx]
        return data

    def randN(self, num):
        for i in range(num):
            yield self.randOne()

    def label2Data(self, label):
        imgL_name = label['imgL']
        imgS_name = label['imgS']
        imgL_path = Path(self.imgL_path) / imgL_name
        imgS_path = Path(self.imgS_path) / imgS_name
        imgL = cv2.imread(imgL_path.__str__())
        imgS = cv2.imread(imgS_path.__str__())
        imgL = self._trans_img(imgL)
        imgS = self._trans_img(imgS)
        box = label['box']
        box = np.array(box)
        return self._Data(imgL_name, imgS_name, imgL, imgS, box)

    def __getitem__(self, i):
        label = self.labels[i]
        if type(label) is list:
            return [self.label2Data(l) for l in label]
        else:
            return self.label2Data(label)

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    import sys

    sys.path.append('../')  # !!!!!关于导包路径的问题终于搞懂了

    import matplotlib.pyplot as plt
    from PIL import Image
    from PIL.ImageDraw import ImageDraw

    imgL_path = '../data/widerface/imgL'
    imgS_path = '../data/widerface/imgS'
    label_path = '../data/widerface/label.json'

    dataset = Dataset(imgL_path, imgS_path, label_path)

    data = dataset.randOne()
    print(data.box)
    p1 = plt.subplot(311)
    p2 = plt.subplot(312)
    p3 = plt.subplot(313)

    p2.imshow(data.imgS)

    im = Image.fromarray(data.imgL)
    crop = im.crop(data.box)
    p3.imshow(crop)

    imgL = Image.fromarray(data.imgL)
    imgLD = ImageDraw(imgL)
    imgLD.rectangle(data.box.tolist())
    p1.imshow(imgL)

    plt.show()
