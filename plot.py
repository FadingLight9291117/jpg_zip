import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    data_path = 'result/main/face/mae.json'
    data = json.load(open(data_path))
    qualities = data['qualities']
    maes = data['maes']
    qualities = np.array(qualities)
    maes = np.array(maes)
    p1 = plt.subplot(221)
    p2 = plt.subplot(222)
    p3 = plt.subplot(223)
    p1.set_title('min')
    p2.set_title('max')
    p3.set_title('mean')
    p1.plot(qualities, maes.min(axis=1))
    p2.plot(qualities, maes.max(axis=1))
    p3.plot(qualities, maes.mean(axis=1))
    plt.savefig('charts/mmm.png')

if __name__ == '__main__':
    main()
