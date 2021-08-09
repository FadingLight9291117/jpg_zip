from pathlib import Path
import json
from easydict import EasyDict as edict

import cv2
from tqdm import tqdm


def get_crop(img, box):
    return img[box[1]: box[3], box[0]: box[2]]


faceset_path = Path('face')
imgL_path = faceset_path / 'imgL'
imgS_path = faceset_path / 'imgS'
imgS_compressed_path = faceset_path / 'imgS_compressed'

faceset = list(imgL_path.glob('*.jpg'))
faceset = {path.name: path for path in faceset}

label_path = list(imgL_path.glob('*.json'))[0]
with label_path.open() as f:
    labels = json.load(f)

infos = []
# 扣出小图保存，并得到标签
labels = tqdm(labels)
for label in labels:
    label = edict(label)
    imgL_name = label.img_name
    imgL_file = imgL_path / imgL_name
    imgL = cv2.imread(imgL_file.__str__())
    boxes = label.boxes
    for i, box in enumerate(boxes):
        imgS = get_crop(imgL, box)
        imgS_name = f'{Path(imgL_name).stem}_crop_{i}.jpg'
        save_path = imgS_path / imgS_name
        cv2.imwrite(save_path.__str__(), imgS)
        info = {
            'imgL': imgL_name,
            'imgS': imgS_name,
            'box': box,
        }

        infos.append(info)

# 保存标签
with (faceset_path / 'label.json').open('w', encoding='utf-8') as f:
    json.dump(infos, f)
