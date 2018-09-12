import cv2
import h5py
import numpy as np

f = h5py.File('data/fold_1.hdf', 'r')
images = np.array(f.get('images'))
labels = np.array(f.get('labels'))
f.close()

while True:
    i = np.random.randint(len(images))
    label = labels[i]

    print images[i]
    cv2.imshow('%d' % label, images[i] / 255.)
    cv2.waitKey(0)


