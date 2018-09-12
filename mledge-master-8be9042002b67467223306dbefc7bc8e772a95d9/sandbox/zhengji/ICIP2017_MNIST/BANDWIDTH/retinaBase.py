LOAD_FROM_HDF = 0
LOAD_FROM_JPG = 1

LOAD_TRAIN = 0
LOAD_VALID = 1
LOAD_TEST  = 2

import h5py
import os
import numpy as np
import collections

class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, distort):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        result_images = self._images[start:end]

        return result_images, self._labels[start:end]


class Reader(object):
    def __init__(self, nodepath, dirname):
        self._nodepath = nodepath
        self._dirname  = dirname
        train_images, train_labels = self.load(LOAD_FROM_HDF, LOAD_TRAIN)
        test_images,  test_labels  = self.load(LOAD_FROM_HDF, LOAD_TEST)

        self.train = DataSet(train_images, train_labels)
        self.test  = DataSet(test_images,  test_labels)
    
    def load(self, mode, subset=None):
        assert (subset == LOAD_TRAIN) or (subset == LOAD_VALID) or (subset == LOAD_TEST), (
            'Plase indicate subset TRAIN/VALID/TEST')

        data_indicator = ['train', 'valid', 'test'][subset]

        if mode == LOAD_FROM_HDF:
            print self._nodepath, self._dirname, data_indicator
            filepath = os.path.join(self._nodepath, self._dirname, data_indicator + '.hdf')
            assert os.path.exists(filepath),(
                "file " + filepath + " is not found in node: " + self._nodepath)
            print 'loading data from '+filepath + '...'
            h5 = h5py.File(filepath ,'r')
            images = np.array(h5.get('images'))
            labels = np.array(h5.get('labels'))
            h5.close()
            print 'images: %s | labels: %s' % (images.shape, labels.shape)

        elif mode == LOAD_FROM_JPG:
            raise NotImplemented  

        return images, labels
