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
 
def cifar_generator(filenames, batch_size, data_dir,site=3):

    all_data = []
    all_labels = []
    if site <0:
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
        #print labels.shape
   
    # nSamples = len(all_data)
    if site == 0:
        all_data = np.load(data_dir+'/data0.npy')
        all_labels = np.load(data_dir+'/label0.npy')
        images = all_data
        labels = all_labels
    if site == 1:
        all_data = np.load(data_dir+'/data1.npy')
        all_labels = np.load(data_dir+'/label1.npy')
        images = all_data
        labels = all_labels
   
    
 
   # print site
 
    print 'label shape:',np.asarray(labels).shape
 
 
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
 
        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])
 
    return get_epoch
 
 
def load(batch_size, data_dir,site,test_dir = None,testing=False):
    if testing:
        return(
            cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir,site),            
            cifar_generator(['test_batch'], batch_size, test_dir,-1)
            )
    else:
        return (
            cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir,site),
            cifar_generator(['test_batch'], batch_size, data_dir,-1)
        )