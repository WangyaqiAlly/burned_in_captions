import numpy as np

import os
import urllib
import gzip
import cPickle as pickle
import cv2

from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Crop(px=(0, 3)), # crop images from each side by 0 to 4px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    #iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

AUGMENTATION=True
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']



def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image

def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch



def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    #images = images[0:1000,:]
    #labels = labels[0:1000]

    def get_epoch():
        print 'start new unlabeded epoch,shuffling...'
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def cifar_generator2(filenames, batch_size, data_dir,n_examples):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    images = images[0:n_examples,:]
    labels = labels[0:n_examples]

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def cifar_generator_lab( batch_size, data_dir,labeled_size):
    all_data = []
    all_labels = []
    lab_data = []
    lab_labels = []
    # site_num=int(labeled_size/1250)
    # for site_i in xrange(site_num):
    #
    #     #data, labels = unpickle(data_dir + '/' + filename)
    #     data=np.load(os.path.join(data_dir,'site{}_1250_data.npy'.format(site_i)))
    #     labels=np.load(os.path.join(data_dir,'site{}_1250_label.npy'.format(site_i)))
    #     print 'loading data from site{} ..........'.format(site_i)
    #     all_data.append(data)
    #     all_labels.append(labels)
    site_num = int(labeled_size / 2500)
    for site_i in xrange(site_num):

        #data, labels = unpickle(data_dir + '/' + filename)
        data=np.load(os.path.join(data_dir,'site{}_2500_data.npy'.format(site_i)))
        labels=np.load(os.path.join(data_dir,'site{}_2500_label.npy'.format(site_i)))
        print 'loading data from site{} ..........'.format(site_i)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    images=(images+0.5)*255.0

    # for label_i in xrange(10):
    #     label_i_ind=np.where(labels==label_i)[0][:labeled_size/10]
    #     print label_i_ind
    #
    #     lab_data.append(images[label_i_ind])
    #     lab_labels.append(labels[label_i_ind])

    #images = np.concatenate( lab_data, axis=0)
    #labels = np.concatenate(lab_labels, axis=0)
    print "labeled train data loaded :", images.shape, labels.shape
    #if AUGMENTATION:
    print 'augmenting...'
    #images = np.reshape(images, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    print images.shape,images.max(),images.min()
    # padding_size = 2
    # pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    # images = np.pad(images, pad_width=pad_width, mode='constant', constant_values=0)


    def get_epoch():
        print 'start new labeded epoch,shuffling...'

        # if AUGMENTATION:
        #images = random_crop_and_flip(images, 2)
        #images_a = np.reshape(images, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)

        # cropped_batch = np.zeros(len(images) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        #     len(images), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
        # for i in range(len(images)):
        #     x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        #     y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        #     cropped_batch[i, ...] = images[i, ...][x_offset:x_offset + IMG_HEIGHT,
        #                             y_offset:y_offset + IMG_WIDTH, :]
        #
        #     cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
        # cropped_batch= np.reshape(cropped_batch.transpose(0, 3, 1, 2), (5000, 3072))

        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        images_aug = seq.augment_images(images)

        images_aug = images_aug.transpose(0, 3, 1, 2).reshape((-1,3072))
        #print images_aug.shape, images_aug.max(), images_aug.min()
        for i in xrange(len(images_aug) / batch_size):
            yield (images_aug[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])
            #assert 0
    return get_epoch





def load(batch_size, data_dir,n_examples):
    return (
        cifar_generator2(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir,n_examples), 
        cifar_generator(['test_batch'], batch_size, data_dir)
    )

def load_semi(batch_size,labeled_data_dir, data_dir,labeled_size):
    return (
        cifar_generator_lab( batch_size, labeled_data_dir,labeled_size),
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir),
        cifar_generator(['test_batch'], batch_size, data_dir)
    )