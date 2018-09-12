LOAD_FROM_HDF = 0
LOAD_FROM_JPG = 1

LOAD_TRAIN = 0
LOAD_VALID = 1
LOAD_TEST  = 2

import h5py
import os
import numpy as np
import collections
from DataAugmentation import *
from multiprocessing import Process, Lock, Event, Value, Queue
import time

NUM_PROCESSES = 20
MAX_QUEUE = 10000

class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def image_worker(self, worker_num, q, e, work_queue):
        while True:
            try:
                image, label = work_queue.get(timeout=0.1)
                image = random_crop(image)
                image = random_HLS(image)
                image = random_flip(image)

                q.put([image, label])
            except:
                if e.is_set():
                    return



    def loop_fill(self, epoch_num, batch_size, q, e):
	self.workers = []
        self.pipequeue = Queue()

        for k in range(NUM_PROCESSES):
            worker = Process(target = self.image_worker, args=(k, q, e, self.pipequeue))
            worker.daemon = True
            worker.start()
            self.workers += [worker]

        for i in xrange(epoch_num * (self.num_examples / batch_size)):
        #for i in range(1):
            #print 'processing batch#', i
            while (self.pipequeue.qsize() > MAX_QUEUE) or (q.qsize() > MAX_QUEUE):
                time.sleep(0.2)
	    self.batch_images, self.batch_labels = self.next_batch(batch_size)
            for image, labels in zip(self.batch_images, self.batch_labels):
                self.pipequeue.put([image, labels])


        for worker in self.workers:
            worker.join()

    def start_fill(self, epoch_num, batch_size):
	print 'start filling...'
	self._resource_pool = Queue()
        self._stop_event = Event()
	self._filler = Process(target = self.loop_fill, args=(epoch_num, batch_size, self._resource_pool, self._stop_event))
	self._filler.start()
        print 'start_fill finish.'

    def stop_fill(self):
	print 'stop filling.'
        self._stop_event.set()
        self._filler.join()
        print 'filling stoped'

    def next_batch_from_queue(self, batch_size):
        ready_images = []
        ready_labels = []
        for i in range(batch_size):
            image, label = self._resource_pool.get()
            ready_images += [image]
            ready_labels += [label]
        return ready_images, ready_labels
    
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
        
    def next_batch(self, batch_size, distort=False):
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
        #if distort == True:
        #    result_images = random_crop(result_images)
        #    result_images = random_HLS(result_images)
        #    result_images = random_flip(result_images)
        
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
