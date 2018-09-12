import numpy as np

import os
import urllib
import gzip
import cPickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    
    fo.close()
    return dict['data'], dict['labels']

def cifar_data(filenames, data_dir, site=3, rate=1.0):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)
    
    nSamples = len(all_data)
    if site == 0:
        all_data = all_data[0:nSamples:2]
        all_labels = all_labels[0:nSamples:2]
    if site == 1:
        all_data = all_data[1:nSamples:2]
        all_labels = all_labels[1:nSamples:2]

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    ind = np.where(np.random.binomial(1, rate, len(images)))[0]
    
    images = images[ind]
    labels = labels[ind]

    return images, labels


def cifar_generator(filenames, batch_size, data_dir, site=3, rate=1.0):

    images, labels = cifar_data(filenames, data_dir, site, rate)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir,site,rate):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir,site,rate=rate), 
        cifar_generator(['test_batch'], batch_size, data_dir,-1)
    )

def cifar_reconstructor_base(data_dir, samples_per_class=10):
    images, labels = cifar_data(['data_batch_1'], data_dir)

    imgs = None
    classes = np.unique(labels)
    # For each class, choose `samples_per_class` images to show
    for _class in classes:
        one_class_all_ids = np.where(labels == _class)[0]
        if len(one_class_all_ids) != 0:
            one_class_num_ids = np.random.choice(one_class_all_ids, samples_per_class)  # .tolist()
            if imgs is None:
                imgs = images[one_class_num_ids]
            else:
                imgs = np.concatenate([imgs, images[one_class_num_ids]])

    return imgs


