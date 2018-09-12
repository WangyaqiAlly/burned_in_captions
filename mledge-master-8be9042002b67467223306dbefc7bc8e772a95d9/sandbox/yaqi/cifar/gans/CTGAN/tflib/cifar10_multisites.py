import numpy as np

import os
import urllib
import gzip
import cPickle as pickle
import tensorflow as tf

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']


def cifar_generator(filenames, batch_size, data_dir, sites='test'):
    all_data = []
    all_labels = []
    if sites == 'test':
        for filename in filenames:
            data, labels = unpickle(data_dir + '/' + filename)
            all_data.append(data)
            all_labels.append(labels)
            print 'running'
        print 'running +++++++++++++++++++++++++'
        images = np.concatenate(all_data, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        # images = np.concatenate(all_data, axis=0)
        # labels = np.concatenate(all_labels, axis=0)
        # print labels.shape

    # nSamples = len(all_data)
    #if site == 0:
    else:
        for i in xrange(len(sites)):
            tf.logging.info("loading 4_sites_data{}.npy".format(sites[i]))
            data_site = np.load(os.path.join(data_dir,'4_sites_data{}.npy'.format(sites[i])))
            labels_site = np.load(os.path.join(data_dir,'4_sites_label{}.npy'.format(sites[i])))

            if i == 0:
                all_data = data_site
                all_labels = labels_site
            else:
                all_data=np.concatenate((all_data,data_site))
                all_labels=np.concatenate((all_labels,labels_site))

    # if site == 1:
    #     all_data = np.load('data1.npy')
    #     all_labels = np.load('label1.npy')


        images = all_data
        labels = all_labels

    tf.logging.info('Loading site:{}, shape:{}'.format(sites[:],images.shape))
    print 'shuffling...'
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    def get_epoch():
        print 'strat new epoch,shuffling...'
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    return get_epoch

def cifar_generator_lab( batch_size, data_dir,labeled_size,sites):
    all_data = []
    all_labels = []
    for site_i in sites:
        data=np.load(os.path.join(data_dir,'site{}_{}_data.npy'.format(site_i,labeled_size)))
        labels=np.load(os.path.join(data_dir,'site{}_{}_label.npy'.format(site_i,labeled_size)))
        tf.logging.info('loading data from site{} ..........'.format(site_i))
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    images=(images+0.5)*255.0

    tf.logging.info("labeled train data loaded :{}, {}".format(images.shape, labels.shape))

    tf.logging.info("{},{},{}".format(images.shape,images.max(),images.min()))


    def get_epoch():


        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)


        images = images.transpose(0, 3, 1, 2).reshape((-1,3072))
        #print images_aug.shape, images_aug.max(), images_aug.min()
        for i in xrange(len(images) / batch_size):
            yield (images_aug[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])
    return get_epoch



def load(batch_size, data_dir, sites):
    return (
        cifar_generator(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'], batch_size,
                        data_dir, sites),
        cifar_generator(['test_batch'], batch_size, '/home/yaqi/Documents/Data/cifar10/cifar-10-batches-py', 'test')
    )

def load_semi(batch_size,labeled_data_dir, data_dir,labeled_size,sites):
    return (
        cifar_generator_lab( batch_size, labeled_data_dir,labeled_size,sites),
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir,sites),
        cifar_generator(['test_batch'], batch_size, data_dir, 'test')
    )