import numpy as np

import os
import urllib
import gzip
import cPickle as cPickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir,site=3, rate=1.0):
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
    ind = np.where(np.random.binomial(1,rate,len(images)))[0]
    
    images = images[ind]
    labels = labels[ind]

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch



def load_imgnet5_train(datadir='./data',shuffle=True,nt=4,h=32,w=32,depth=3):
    print 'load training data...'
    d = np.array([]).reshape([0, h, w, depth])
    l = np.array([]).astype(int)
    for k in range(nt):
    #for k in range(1):
        dfilename = 'train-images-%02d.pickle' % k
        lfilename = 'train-labels-%02d.pickle' % k
        dfile = os.path.join(datadir, dfilename)
        lfile = os.path.join(datadir, lfilename)

        with open(dfile, 'rb') as f:
            dtmp = cPickle.load(f)
        f.close()

        with open(lfile, 'rb') as f:
            ltmp = (cPickle.load(f)).astype(int)
        f.close()

        # Concatenate along axis 0 by default
        d = np.concatenate((d, dtmp))
        l = np.concatenate((l, ltmp))

    if shuffle:
        d, l = rshuffle(d, l)
    print 'training: d.shape: ', d.shape, ' l.shape: ', l.shape, 'd.max:', d.max(),' d.min: ',d.min(),'\n ',\
                         'l.max:', l.max(),' l.min: ', l.min()
    return d, l


def load_imgnet5_val(datadir='./data', shuffle=True):
    print 'load validation data...'
    dtfile = os.path.join(datadir, 'test-images.pickle')
    ltfile = os.path.join(datadir, 'test-labels.pickle')
    with open(dtfile, 'rb') as f:
        dt = cPickle.load(f)
    f.close()
    with open(ltfile, 'rb') as f:
        lt = (cPickle.load(f)).astype(int)
    f.close()
    'random shuffle training & validation data once'
    if shuffle:
        dt, lt = rshuffle(dt, lt)

    return dt, lt

def imgnet5_generator(filename, batch_size, rate=1.0,trainval='train'):
    all_data = []
    all_labels = []
    if trainval is 'train':
        images, labels= load_imgnet5_train(filename,False,4)
    else:
        images, labels = load_imgnet5_val(filename, False)
    #images = np.load(filename+'images.npy')
    images = images[..., [2, 1, 0]]
    images = np.transpose(images, (0,3,1,2))
    images = np.reshape(images, (-1, 3072))

    #labels = np.load(filename+'labels.npy')
    print images.shape
    #d = images / 255.  # [..., [2, 1, 0]]/255.
    #all_data = list(d)
    #nSamples = len(all_data)

    #images = np.concatenate(all_data, axis=0)
    #labels = np.concatenate(all_labels, axis=0)
    ind = np.where(np.random.binomial(1, rate, len(images)))[0]

    images = images[ind]
    labels = labels[ind]

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    return get_epoch

def load_imgnet5(data_dir,batch_size,rate):
    return (
        imgnet5_generator(data_dir, batch_size, rate=rate,trainval='train'),
        imgnet5_generator(data_dir,  batch_size,1.0,trainval='val')
    )



    
    

def load(batch_size, data_dir,site,rate):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir,site,rate=rate), 
        cifar_generator(['test_batch'], batch_size, data_dir,9999)
    )