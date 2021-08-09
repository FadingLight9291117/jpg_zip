from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

label_file = 'widerface/label.json'

labels = json.load(open(label_file))

sizes = []

for label in labels:
    box = label['box']
    w = box[2] - box[0]
    h = box[3] - box[1]
    sizes.append(w * h)

sizes = np.array(sizes)

sizes = np.sort(sizes)

print(len(sizes))
plt.plot(sizes)
plt.show()
