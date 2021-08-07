import json
from pathlib import Path

from PIL import Image, ImageDraw
from easydict import EasyDict as edict
from tqdm import tqdm


def wh2bos(wh_box):
    return [wh_box[0], wh_box[1], wh_box[0] + wh_box[2], wh_box[1] + wh_box[3]]


label_path = 'faceset/images/wh_label.json'
newlabel_path = 'faceset/images/label.json'

with open(label_path) as f:
    labels = json.load(f)

for label in labels:
    wh_boxes = label['boxes']
    boxes = [wh2bos(wh_box) for wh_box in wh_boxes]
    label['boxes'] = boxes

with open(newlabel_path, 'w', encoding='utf-8') as f:
    json.dump(labels, f)
