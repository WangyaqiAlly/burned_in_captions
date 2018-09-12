import os
import h5py
import numpy as np
import cv2

imagenet_dir = '/Data/imagenet-10-32x32-fold'
train_dir = 'train'
validation_dir = 'validation'
site_num = 5
time_num = 10
class_num = 10


print 'Dealing with training data:'
for t in range(1, time_num+1):
    f = h5py.File(os.path.join('data', 'fold_%d.hdf' % (t)), 'w')
    image_num = 0
    for s in range(1, site_num+1):
        for c in range(1, class_num+1):
            path = os.path.join(imagenet_dir, train_dir, 'site-%d' % (s), 'time-%02d' % (t), 'class-%02d' % (c))
            image_num += len(os.listdir(path))

    images = np.zeros((image_num, 32, 32, 3))
    labels = np.zeros((image_num), dtype=np.int)

    index = 0
    for s in range(1, site_num+1):
        for c in range(1, class_num+1):
            path = os.path.join(imagenet_dir, train_dir, 'site-%d' % (s), 'time-%02d' % (t), 'class-%02d' % (c))
            for filename in os.listdir(path):
                image = cv2.imread(os.path.join(path, filename))
                label = c-1
                assert(label >= 0)
                images[index] = image
                labels[index] = label
                index += 1
    f.create_dataset('images', data = images, dtype = 'f')
    f.create_dataset('labels', data = labels, dtype = 'i')
    f.close()
    print 'Finished fold %d with %d smaples.' % (t, index)

print 'Dealing with validation data:'
image_num = 0 
f = h5py.File(os.path.join('data', 'test.hdf'), 'w')
for c in range(1, class_num+1):
    path = os.path.join(imagenet_dir, validation_dir, 'class-%02d' % (c))
    image_num += len(os.listdir(path))

images = np.zeros((image_num, 32, 32, 3))
labels = np.zeros((image_num), dtype=np.int)

index = 0
for c in range(1, class_num+1):
    path = os.path.join(imagenet_dir, validation_dir, 'class-%02d' % (c))
    for filename in os.listdir(path):
        image = cv2.imread(os.path.join(path, filename))
        label = c-1
        assert(label >= 0)
        images[index] = image
        labels[index] = label
        index += 1
f.create_dataset('images', data = images, dtype = 'f')
f.create_dataset('labels', data = labels, dtype = 'i')
f.close()
print 'Finished testset with %d smaples.' % (index)
        


