from pathlib import Path
import json

import cv2

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

for label in labels:
    im = cv2.imread()
