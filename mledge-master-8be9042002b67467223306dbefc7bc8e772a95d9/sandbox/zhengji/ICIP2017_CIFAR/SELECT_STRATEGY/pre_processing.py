import random
import numpy as np
from data_utils_CIFAR10 import load_CIFAR10
import matplotlib.pyplot as plt
import h5py
import os
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# Load the raw CIFAR-10 data.
cifar10_dir = '../CIFAR10_data'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

print 'shuffling...'
perm = np.arange(X_train.shape[0])
np.random.shuffle(perm)
X_train = X_train[perm]
y_train = y_train[perm]

'''
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

'''
ratio = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
name  = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'fold_6', 'fold_7', 'fold_8', 'fold_9']
total_num = X_train.shape[0]
last = 0
for r, n in zip(ratio, name):
    print 'processing from %d to %d.' % (last, last + int(total_num * r))
    train_images = X_train[last : last + int(total_num * r)]
    train_labels = y_train[last : last + int(total_num * r)]
    last = last + int(total_num * r)
    if not os.path.exists('data'):
        os.mkdir('data')
    f = h5py.File(os.path.join('data', n+'.hdf'), 'w')
    f.create_dataset('images', data = train_images, dtype = 'f')
    f.create_dataset('labels', data = train_labels, dtype = 'i')
    f.close()

f = h5py.File(os.path.join('data','test.hdf'), 'w')
f.create_dataset('images', data = X_test, dtype = 'f')
f.create_dataset('labels', data = y_test, dtype = 'i')
f.close()
