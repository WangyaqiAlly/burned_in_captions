import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import h5py
import os
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# Load the raw CIFAR-10 data.
mnist_dir = 'MNIST_data/'
mnist = input_data.read_data_sets(mnist_dir, one_hot=False)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test  = mnist.validation.images
y_test  = mnist.validation.labels

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

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
'''
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow((255*X_train[idx]).astype('uint8').reshape((28,28)))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
'''

#ratio = [0.2, 0.2, 0.2, 0.2, 0.2]
#name  = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
fold_num = 11
total_num = X_train.shape[0]
last = 0
for i in range(fold_num):
    print 'processing from %d to %d.' % (last, last + int(total_num / fold_num))
    train_images = X_train[last : last + int(total_num / fold_num)]
    train_labels = y_train[last : last + int(total_num / fold_num)]
    last = last + int(total_num / fold_num)
    if not os.path.exists('data'):
        os.mkdir('data')
    f = h5py.File(os.path.join('data', 'fold_'+str(i+1)+'.hdf'), 'w')
    f.create_dataset('images', data = train_images, dtype = 'f')
    f.create_dataset('labels', data = train_labels, dtype = 'i')
    f.close()

f = h5py.File(os.path.join('data','test.hdf'), 'w')
f.create_dataset('images', data = X_test, dtype = 'f')
f.create_dataset('labels', data = y_test, dtype = 'i')
f.close()
