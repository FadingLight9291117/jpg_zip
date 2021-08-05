from pathlib import Path
import json
from easydict import EasyDict as edict

import cv2


def get_crop(img, box):
    return img[box[1]: box[3], box[0]: box[2]]


faceset_path = Path('faceset')
images_path = faceset_path / 'images'
imgL_path = faceset_path / 'imgL'
imgS_path = faceset_path / 'imgS'
imgS_compressed_path = faceset_path / 'imgS_compressed'

faceset = list(images_path.glob('*.jpg'))
faceset = {path.name: path for path in faceset}

label_path = list(images_path.glob('*.json'))[0]
with label_path.open() as f:
    labels = json.load(f)

infos = []

for label in labels:
    label = edict(label)
    imgL_name = label.img_name
    img_path = imgL_path / imgL_name
    imgL = cv2.imread(img_path.__str__())
    boxes = label.boxes
    for i, box in enumerate(boxes):
        imgS = get_crop(imgL, box)
        imgS_name = f'{Path(imgL_name).stem}_{i}.jpg'
        save_path = imgS_path / imgS_name
        try:
            cv2.imwrite(save_path.__str__(), imgS)
        except Exception:
            ...
        info = {
            'imgL_name': imgL_name,
            'imgS_name': imgS_name,
            'box': box,
        }
        infos.append(info)
with (faceset_path / 'img_info.json').open('w', encoding='utf-8') as f:
    json.dump(infos, f)
