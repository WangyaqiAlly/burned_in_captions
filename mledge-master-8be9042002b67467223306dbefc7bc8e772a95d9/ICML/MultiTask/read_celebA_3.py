################################
#  make num example smaller to see if memory problem goes away

import numpy as np
import os
import cv2
from celebLabel import AttributesCelebA
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def populate_dir(the_dir, labelArray):
    fwidth  = 64
    fheight = 64

    files = os.listdir(the_dir)
    data  = np.zeros((len(files), 3, fwidth,fheight))
    label = np.zeros((len(files)), dtype='int32')

    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(the_dir, file), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (fwidth,fheight), interpolation = cv2.INTER_LANCZOS4)
        img = img.transpose(2,0,1)
        tokens = file.split('.')
        idx = int(tokens[0])-1
        if labelArray is None:
            label[i] = 0
        elif labelArray[idx] < 0:
            label[i] = 0
        else:
            label[i] = 1
        data[i] = img

    data = data.reshape(len(files), 3*fwidth*fheight).astype('int32')
    return data, label

def read_data(labelNames, train_dir, test_dir, labelfile):
    labelobj   = AttributesCelebA(labelfile)
    labelarray = labelobj.getLabels(labelNames)

    test_data,  test_label  = populate_dir(test_dir, labelarray)
    train_data, train_label = populate_dir(train_dir,labelarray)

    return train_data, train_label, test_data, test_label


def celeb_generator(path, labelarray, batch_size):
    images, labels = populate_dir(path, labelarray)
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch

#############
#
#   for integration with CT-GAN

def read_data_semi(labelNames, train_unlabelled_dir, train_labelled_dir, test_dir, labelfile, batch_size):
    labelObj   = AttributesCelebA(labelfile)
    labelarray = labelObj.getLabels(labelNames)

    return (
        celeb_generator(train_labelled_dir,   labelarray, batch_size),
        celeb_generator(train_unlabelled_dir, None,       batch_size),
        celeb_generator(test_dir,             labelarray, batch_size)
    )


########################################################################
#
#   for integration with VAT

from dataset_utils import *

#############
#
#   makes the tfrecord if not already exist

g_labelObj = None

def format_for_vat(data, label):
    data = data.astype(np.float32)
    data = (data - 127.5)/255.
    label = label.astype(np.int64)
    return data, label

def derive_input_dir_name(labelname, ganlabel, label_src='unused'):
    labelled   = label_src
    unlabelled = 'train-unlabelled'
    test       = 'test'
    if ganlabel is not None:
        unlabelled = 'generated_' + ganlabel
    return labelled, unlabelled, test

def derive_names(labelname, ganlabel, label_src='unused'):
    tfrecord_labelled = label_src + '_' + labelname + '.tfrecord'
    #if use_all_labelled:
    #    tfrecord_labelled = 'labelled_train_' + labelname + '.tfrecord'
    #else:
    #    tfrecord_labelled   = 'transmitted_'   + labelname + '.tfrecord'
    if ganlabel is None:
        tfrecord_unlabelled = 'unlabelled_train_' + labelname + '.tfrecord'
    else:
        tfrecord_unlabelled = 'generated_' + ganlabel + '.tfrecord'
    tfrecord_test       = 'test_'             + labelname + '.tfrecord'
    return tfrecord_labelled, tfrecord_unlabelled, tfrecord_test

def ensure_one_tfrecord(labelname, dataset_path, input_dir, tfrecord_name, isUnlabelled=False):
    global g_labelObj
    if input_dir is not None:
        if not os.path.isfile(os.path.join(dataset_path, tfrecord_name)):
            print('Generating', tfrecord_name, '...')
            if isUnlabelled:
                labelarray = None
            else:
                if g_labelObj is None:
                    g_labelObj = AttributesCelebA(os.path.join(dataset_path, 'list_attr_celeba.txt'))
                labelarray = g_labelObj.getLabels(labelname)
            data, label = populate_dir(os.path.join(dataset_path, input_dir), labelarray)
            data, label = format_for_vat(data, label)
            convert_images_and_labels(data, label, os.path.join(dataset_path, tfrecord_name))

def inputs(labelname, ganlabel, batch_size=100,
           train=True, validation=False,
           shuffle=True, num_epochs=None, label_src=None):

    dataset_path = '/home2/dataset/Celeb-A'
    input_labelled, input_unlabelled, input_test = derive_input_dir_name(labelname, ganlabel, label_src)
    tfrecord_labelled, tfrecord_unlabelled, tfrecord_test = derive_names(labelname, ganlabel, label_src)

    if train:
        filenames = []
        #nHalfPercent = int(FLAGS.percent_label/0.49999)
        #for i in range(nHalfPercent):
        #    tag = '-' + str(i*0.5)
        ensure_one_tfrecord(labelname, dataset_path, input_labelled, tfrecord_labelled)
        filenames.append(tfrecord_labelled)
        #files = os.listdir(os.path.join(dataset_path, 'train-labelled'))
        num_examples = 400
    else:
        ensure_one_tfrecord(labelname, dataset_path, input_test, tfrecord_test)
        filenames = [tfrecord_test]
        #files = os.listdir(os.path.join(dataset_path, 'test'))
        num_examples = 100
    print('Reading', filenames, '...')
    filenames = [os.path.join(dataset_path, filename) for filename in filenames]

    filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs)
    image, label = read(filename_queue)
    image = transform(tf.cast(image, tf.float32)) if train else image
    return generate_batch([image, label], num_examples, batch_size, shuffle)

def unlabeled_inputs(labelname, ganlabel, batch_size=100,
                     validation=False,
                     shuffle=True):
    dataset_path = '/home2/dataset/Celeb-A'
    input_labelled, input_unlabelled, input_test = derive_input_dir_name(labelname, ganlabel)
    tfrecord_labelled, tfrecord_unlabelled, tfrecord_test = derive_names(labelname, ganlabel)

    ensure_one_tfrecord(labelname, dataset_path, input_unlabelled, tfrecord_unlabelled, isUnlabelled=True)
    print('Reading', tfrecord_unlabelled, '...')
    filenames = [tfrecord_unlabelled]
    #files = os.listdir(os.path.join(dataset_path, 'train-unlabelled'))
    num_examples = 400

    filenames = [os.path.join(dataset_path, filename) for filename in filenames]

    filename_queue = tf.train.string_input_producer(filenames)
    image, label = read(filename_queue)
    image = transform(tf.cast(image, tf.float32))
    return generate_batch([image], num_examples, batch_size, shuffle)


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = read_data('Smiling', '/home2/dataset/Celeb-A/train', '/home2/dataset/Celeb-A/test', '/home2/dataset/Celeb-A/list_attr_celeba.txt')
    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

