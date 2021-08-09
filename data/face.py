import json
from pathlib import Path
import shutil

from PIL import Image

img_dir = Path('face/imgL')

label_path = img_dir / 'label.json'

labels = json.load(label_path.open())

for i, label in enumerate(labels):
    img_name = label['img_name']
    img_path = img_dir / img_name
    shutil.move(img_path.__str__(), img_path.with_name(f'{i}.jpg'))
    label['img_name'] = f'{i}.jpg'

with open('face/imgL/label.json', 'w', encoding='utf-8') as f:
    json.dump(labels, f)
