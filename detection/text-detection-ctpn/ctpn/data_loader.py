

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from scipy import linalg
import glob
import pickle
from PIL import Image
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf

from dataset_utils import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', '../../data/tfdataset/ori_size/',
                           'where to store the dataset')
tf.app.flags.DEFINE_string('test_data_path', '../../data/tfdataset/ori_size/',
                           'where to store the dataset')
tf.app.flags.DEFINE_string('network', 'ctpn','which network')
tf.app.flags.DEFINE_integer('train_size', 30000, "The number of training examples{52160}")
tf.app.flags.DEFINE_integer('test_size', 13040, "The number of validation examples")
tf.app.flags.DEFINE_integer('cls_num', 2,'how many classes')
tf.app.flags.DEFINE_integer('img_width',640,'')
tf.app.flags.DEFINE_integer('img_height',480,'')



def inputs(batch_size=100,
           train=True,
           shuffle=True, num_epochs=None):
    if train:
        filenames = sorted(glob.glob(os.path.join(FLAGS.train_data_path,'*', 'train*')))
        # print(filenames)
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        example_list = [read(filename_queue, train, False,i) for i in  range(len(filenames))]
        # image, label = read(filename_queue, train)
        num_examples = int(FLAGS.train_size)
        # print(example_list)
        tf.logging.info('train inputs: building queue from :{}....'.format(filenames))
        tf.logging.info("image shape:{} , label shape:{}".format(example_list[0][0].shape, example_list[0][1].shape))
        return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                           shuffle=shuffle)
    else:
        filenames = sorted(glob.glob(os.path.join(FLAGS.test_data_path,'*', 'test*')))
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        example_list = [read(filename_queue, train, False,i) for  i in range(len(filenames))]
        # image, label = read(filename_queue, train)
        num_examples = int(FLAGS.test_size)
        tf.logging.info('test inputs: building queue from :{}....'.format(filenames))
        tf.logging.info("image shape:{} , label shape:{}".format(example_list[0][0].shape, example_list[0][1].shape))
        return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                   shuffle=shuffle)



def inputs_ctpn(batch_size=100,
           train=True,
           shuffle=True, num_epochs=None):
    if train:
        filenames = sorted(glob.glob(os.path.join(FLAGS.train_data_path,'*', 'train*')))
        # print(filenames)
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        example_list = [read(filename_queue, train, False,i) for i in  range(len(filenames))]
        # image, label = read(filename_queue, train)
        num_examples = int(FLAGS.train_size)
        # print(example_list)
        tf.logging.info('train inputs: building queue from :{}....'.format(filenames))
        tf.logging.info("image shape:{} , label shape:{}".format(example_list[0][0].shape, example_list[0][1].shape))
        return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                           shuffle=shuffle)
    else:
        filenames = sorted(glob.glob(os.path.join(FLAGS.test_data_path,'*', 'test*')))
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        example = [read(filename_queue, train, False,0)]
        # image, label = read(filename_queue, train)
        num_examples = int(FLAGS.test_size)
        tf.logging.info('test inputs: building queue from :{}....'.format(filenames))
        # tf.logging.info("image shape:{} , label shape:{}".format(example_list[0][0].shape, example_list[0][1].shape))
        return generate_batch(example=example,  dataset_size=num_examples, batch_size=batch_size,
                                   shuffle=shuffle)


def main(argv):
    swd = '../data/test/'
    print("testing here!")
    with tf.device("/cpu:0"):
        with tf.Session() as sess:

            images, labels,langs = inputs(batch_size=100,train=True,shuffle=True, num_epochs=None)
            # images_test, labels_test,langs_test= inputs(batch_size=100,train=False,shuffle=True, num_epochs=None)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            ls = []

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(1):
                _images, _labels,_langs = sess.run([images, labels,langs])
                print(_images.shape, _labels,_langs)
                # _images_test, _labels_test,_langs_test = sess.run([images_test, labels_test,langs_test])
                j=0
                ls_train = []
                for single, l ,tt in zip(_images, _labels,_langs):

                    single =(single +1.0)/2.0*255.0
                    single = single.astype(np.uint8)
                    img = Image.fromarray(single, 'RGB')
                    print(single.shape, single.max(),single.min())
                    # l = np.argmax(l)
                    ls_train.append(tt)
                    img.save(swd + 'train_'+str(i)+'_' +str(j)+ '_label_' + str(l) + '.jpg')
                    j+=1
                    print(j)
                #     ls_train.append(tt)
                #
                # j=0
                # ls =[]
                # for single, l,tt in zip(_images_test, _labels_test,_langs_test):
                #     img = Image.fromarray(single, 'RGB')
                #     single = (single + 1.0) / 2.0 * 255.0
                #     single = single.astype(np.uint8)
                #     img = Image.fromarray(single, 'RGB')
                #     print(single.shape, single.max(), single.min())
                #     # l = np.argmax(l)
                #     ls.append(tt)
                #     img.save(swd + 'test_' + '_' + str(j) + '_label_' + str(l) + '.jpg')
                #     j += 1
                #     print(j)
                # print(ls_train)
                # print(ls)

            # while True:
            #     try:
            #         # image_down = np.asarray(image_down.eval(), dtype='uint8')
            #         # plt.imshow(image.eval())
            #         # plt.show()
            #         _images,_labels= sess.run([images, labels])
            #         j=0
            #         for single, l  in zip(_images,_labels):
            #             img = Image.fromarray(single, 'RGB')
            #             # l = np.argmax(l)
            #             ls.append(l)
            #             img.save(swd + 'dataloader_'+str(i)+'_' +str(j)+ '_label_' + str(l) + '.jpg')
            #             j+=1
            #             print(j)
            #     except:
            #         print("finish!")
            #         break
            print(ls_train)
            print(ls.count(1),ls.count(0))
            coord.request_stop()
            coord.join(threads)
        sess.close()



if __name__ == "__main__":
    tf.app.run()

