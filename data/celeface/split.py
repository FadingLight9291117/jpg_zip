import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def wh2box(wh_box):
    return [
        wh_box[0],
        wh_box[1],
        wh_box[0] + wh_box[2],
        wh_box[1] + wh_box[3],
    ]


image_dir = Path('images')
imgL_dir = Path('imgL')
imgS_dir = Path('imgS')
label_path = Path('list_bbox_celeba.txt')
save_path = Path('label.json')

imgL_dir.mkdir(exist_ok=True)
imgS_dir.mkdir(exist_ok=True)

with label_path.open(encoding='utf-8') as f:
    lines = f.readlines()

labels = {}
for line in lines[2:]:
    line = line.split()
    img_name, *box = line
    box = list(map(int, box))
    labels[img_name] = box

new_label = []
for img in image_dir.glob('*.jpg'):
    box = wh2box(labels[str(img.name)])
    label = {
        'img_name': img.name,
        'boxes': [box],
    }
    new_label.append(label)
    print(f'finish -> {img_name}.')

with save_path.open('w', encoding='utf-8') as f:
    json.dump(new_label, f)
