# Copyright 2015 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Routine for decoding the CIFAR-10 binary file format."""

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

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf

from dataset_utils import *

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('labeled_data_dir', '../Data/cifar_nozca/', 'where to store the dataset')
tf.app.flags.DEFINE_string('unlabeled_inputs_dir', '/home/yaqiwang/yaqi@sandbox/distributed_ml/gans/CTGAN/tensorflow_generative_model/CT_generated/multisites/0123_1250x4_batch1/',
                           'where to store the dataset')


#tf.app.flags.DEFINE_integer('dataset_seed', 1, "dataset seed")

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_TRAIN = 50000
NUM_EXAMPLES_TEST = 10000
#NUM_GEN_TRAIN = FLAGS.NUM_GEN_TRAIN


def inputs(batch_size=100,
           train=True,
           shuffle=True, num_epochs=None):

    if train:
       
        if FLAGS.site == 'central' and FLAGS.gantype == 'CTGAN_SEMI':
            filenames = [
                'labeled_cifar_site0_{}.tfrecords'.format(int(FLAGS.num_labeled_examples / 4)), \
                'labeled_cifar_site1_{}.tfrecords'.format(int(FLAGS.num_labeled_examples / 4)),
                'labeled_cifar_site2_{}.tfrecords'.format(int(FLAGS.num_labeled_examples / 4)),
                'labeled_cifar_site3_{}.tfrecords'.format(int(FLAGS.num_labeled_examples / 4)),
                ]
            #    filenames = ['labeled_train_{}.tfrecords'.format(int(FLAGS.num_labeled_examples))]
            filename_queue = generate_filename_queue(filenames, FLAGS.labeled_data_dir, num_epochs)
            example_list = [read(filename_queue, train)
                            for _ in range(4)]
            tf.logging.info('labeled:building queue from:{}....'.format(filenames))
            tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
            num_examples = int(FLAGS.num_labeled_examples)
            return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                       shuffle=shuffle)
        elif FLAGS.site == '0' and FLAGS.gantype == 'CTGAN_SEMI':
            filenames = ['labeled_cifar_site0_{}.tfrecords'.format(FLAGS.num_labeled_examples)]
            #    filenames = ['labeled_train_{}.tfrecords'.format(int(FLAGS.num_labeled_examples))]
            filename_queue = generate_filename_queue(filenames, FLAGS.labeled_data_dir, num_epochs)
            example_list = [read(filename_queue, train)
                            for _ in range(1)]
            tf.logging.info('building queue from:{}....'.format(filenames))
            tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
            num_examples = int(FLAGS.num_labeled_examples)
            return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                       shuffle=shuffle)


    else:


        filenames = ['test_cifar_nozca.tfrecords']
        num_examples = NUM_EXAMPLES_TEST
        filename_queue = generate_filename_queue(filenames, '../Data/cifar_nozca/', num_epochs)
        image, label = read(filename_queue,train)
        print("labeled_cifar_img_shape:", image.shape, type(image))
        # image = transform(tf.cast(image, tf.float32)) if train else image
        return generate_batch([image, label], num_examples, batch_size, shuffle)





def unlabeled_inputs(batch_size=100,
                     shuffle=True):

    if FLAGS.site == 'central' and FLAGS.gantype == 'CTGAN_UNCOND':
            filenames = sorted(glob.glob(os.path.join(FLAGS.unlabeled_inputs_dir, 'CTGAN_uncond_wdtop_50k_batch*')))
            filenames += sorted(glob.glob(os.path.join(FLAGS.labeled_data_dir, '*1250_nozca*')))
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
            example_list = [read(filename_queue, True)
                            for _ in range(14)]
            tf.logging.info('building queue from:{}....'.format(filenames))
            tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
            num_examples=int(FLAGS.num_unlabeled)
            return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                       shuffle=shuffle)
    elif FLAGS.site == 'central' and FLAGS.gantype == 'CTGAN_THEANO':
            filenames = sorted(glob.glob(os.path.join(FLAGS.unlabeled_inputs_dir, 'CT_*')))
            filenames += sorted(glob.glob(os.path.join(FLAGS.labeled_data_dir, '*1250_nozca*')))
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
            example_list = [read(filename_queue, True)
                            for _ in range(len(filenames))]
            tf.logging.info('building queue from:{}....'.format(filenames))
            tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
            num_examples=int(FLAGS.num_unlabeled)
            return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                       shuffle=shuffle)

    elif FLAGS.site == 'central' and FLAGS.gantype == 'CTGAN_SEMI':
            filenames = sorted(glob.glob(os.path.join(FLAGS.labeled_data_dir, 'labeled_cifar_site*_{}.tfrecords'.format(int(
            FLAGS.num_labeled_examples/4)))))
            #filenames += sorted(glob.glob(os.path.join(FLAGS.labeled_data_dir, 'site1_12500_nozca*')))
            filenames += sorted(glob.glob(os.path.join(FLAGS.unlabeled_inputs_dir, 'CTGAN_ac_wdtop*')))
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
            example_list = [read(filename_queue, True)
                            for _ in range(len(filenames))]
            tf.logging.info('building queue from:{}....'.format(filenames))
            tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
            num_examples=int(FLAGS.num_unlabeled)
            return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                       shuffle=shuffle)

    elif FLAGS.site == '0' and FLAGS.gantype == 'CTGAN_SEMI':
            filenames = sorted(glob.glob(os.path.join(FLAGS.labeled_data_dir, 'labeled_cifar_site0_{}.tfrecords'.format(FLAGS.num_labeled_examples))))
            filenames += sorted(glob.glob(os.path.join(FLAGS.unlabeled_inputs_dir, 'CTGAN_ac_wdtop*')))
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
            example_list = [read(filename_queue, True)
                            for _ in range(len(filenames))]
            tf.logging.info('building queue from:{}....'.format(filenames))
            tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
            num_examples=int(FLAGS.num_unlabeled)
            return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                       shuffle=shuffle)




def main(argv):
    prepare_dataset()


if __name__ == "__main__":
    tf.app.run()
