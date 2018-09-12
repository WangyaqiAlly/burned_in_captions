import numpy as np
import os
import struct

from tensorflow.examples.tutorials import mnist

class Dataset(object):
    def __init__(self, images, labels=None):
        self.epoch = -1
        #self.images = images.reshape(images.shape[0], -1)
        self.images = images
        self.labels = labels
        self.num_samples = images.shape[0]
        self.curpos = self.num_samples
        
    def get_batch_in_epoch(self, batch_size):
        return int(self.num_samples/batch_size)
        
    def next_batch(self, batch_size):
        if self.curpos + batch_size > self.num_samples:
            #############################
            #  shuffle and resets...
            self.epoch += 1
            self.curpos = 0
            rng_seed = np.random.get_state()
            np.random.shuffle(self.images)
            if self.labels is not None:
                np.random.set_state(rng_seed)
                np.random.shuffle(self.labels)
        start = self.curpos
        self.curpos += batch_size
        if self.labels is None:
            return self.images[start:self.curpos], None
        else:
            return self.images[start:self.curpos], self.labels[start:self.curpos]

class MnistDataset(Dataset):
    def __init__(self, dir='Mnist'):
        #################################
        #  don't pull, assume rob's script has downloaded the images.
        #
        with open('train-images-idx3-ubyte','rb') as f:
            h = struct.unpack('>IIII',f.read(16))
            d = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
            d = d/255.
            f.close()
        with open('train-labels-idx1-ubyte','rb') as f:
            h = struct.unpack('>II', f.read(8))
            lt = np.fromstring(f.read(), dtype=np.uint8).astype('int32')
            f.close()
        print d.shape
        super(MnistDataset, self).__init__(d,lt)
