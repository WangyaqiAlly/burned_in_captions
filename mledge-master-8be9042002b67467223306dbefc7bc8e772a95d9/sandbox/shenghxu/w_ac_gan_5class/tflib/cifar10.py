import numpy as np

import os
import urllib
import gzip
import cPickle as pickle
import cv2

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir,site=3):
    # all_data = []
    # all_labels = []
    # for filename in filenames:        
    #     data, labels = unpickle(data_dir + '/' + filename)
    #     all_data.append(data)
    #     all_labels.append(labels)
    
    # nSamples = len(all_data)
    # if site == 0:
    #     all_data = all_data[0:nSamples:2]
    #     all_labels = all_labels[0:nSamples:2]
    # if site == 1:
    #     all_data = all_data[1:nSamples:2]
    #     all_labels = all_labels[1:nSamples:2]

    

    # images = np.concatenate(all_data, axis=0)
    # labels = np.concatenate(all_labels, axis=0)

   # print 'orig_image shape', images.shape
    graph_dir = '/home2/shenghxu/image_net_new/dataset-5-class-well'
    train_images = None
    for time in xrange(1, 10+ 1):
        train_images_t, train_labels_t = load_imageNet(graph_dir, 'train', '1', 5)
        if train_images is None:
            train_images, train_labels = train_images_t, train_labels_t
        else:
            train_images = np.concatenate([train_images, train_images_t], axis=0)
            train_labels = np.concatenate([train_labels, train_labels_t], axis=0)

    # print 'image shape', images.shape
    leng = train_images.shape[0]
    images = np.reshape(train_images,(leng,32*32*3))
    #images = np.concatenate(images,axis=0)
    labels = np.reshape(train_labels,leng)
    print labels.max
    print 'data max', images.max()

     
    print 'label shape:', labels.shape
    print 'label max', labels.max()
   

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load_imageNet(graph_dir, flag='train', edge_id='0', time_id=1):
    
    images, labels = None, None
    
    
    num_images = 0
    

    parent_path = os.path.join(graph_dir, 'train', 'site-%s' % edge_id, 'time-%02d' % time_id)
    num_classes = len(os.listdir(parent_path))
    for cls in range(1, num_classes + 1):
        graph_path = os.path.join(parent_path, 'class-%02d' % cls)
        num_images += len(os.listdir(graph_path))

    images = np.zeros((num_images, 32, 32, 3))
    labels = np.zeros(num_images, dtype=np.int)



    index = 0
    for cls in range(1, num_classes + 1):
        graph_path = os.path.join(parent_path, 'class-%02d' % cls)
        for image_name in os.listdir(graph_path):
            image = cv2.imread(os.path.join(graph_path, image_name))
            label = cls - 1
            assert (label >= 0)
            images[index] = image
            labels[index] = label
            index += 1
        
    

    return images, labels

def load(batch_size, data_dir,site):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir,site), 
        cifar_generator(['test_batch'], batch_size, data_dir,9999)
    )