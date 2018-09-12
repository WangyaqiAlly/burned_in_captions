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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', '../../Data/lsun20-tf-balance/',
                           'where to store the dataset')
tf.app.flags.DEFINE_string('test_data_path', '../../Data/lsun20-tf-balance/test/',
                           'where to store the dataset')
#tf.app.flags.DEFINE_string('site', 'site1_test',
#                           'where to store the dataset')
tf.app.flags.DEFINE_integer('percent', 20,
                           'how many percent for training a gan')
tf.app.flags.DEFINE_integer('train_size_per_pct', 60000, "The number of training examples{5037800,6241830 6847288}")
tf.app.flags.DEFINE_integer('test_size', 60000, "The number of validation examples")
tf.app.flags.DEFINE_integer('num_cls', 20,'how many classes {20,10}')

def inputs(batch_size=100,
           train=True,
           shuffle=True, num_epochs=None):
    if train:
        # if FLAGS.site == 'central':
        #     filenames = sorted(glob.glob(os.path.join(FLAGS.train_data_path, '*.tfrecords')))
        #     filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        #     example_list = [read(filename_queue, train)
        #                     for _ in range(10)]
        #     num_examples = FLAGS.train_size_per_site
        #     print('building queue from:{}....'.format(filenames))
        #     print(example_list[0][0].shape, example_list[0][1].shape)
        #     return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
        #                                shuffle=shuffle)
        # if FLAGS.site == 'onesite':
        #     filenames = sorted(glob.glob(os.path.join(FLAGS.train_data_path, 'site0', '*.tfrecords')))
        #     filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        #     example_list = [read(filename_queue, train)
        #                     for _ in range(10)]
        #     num_examples = FLAGS.train_size_per_site
        #     print('building queue from:{}....'.format(filenames))
        #     print(example_list[0][0].shape, example_list[0][1].shape)
        #     return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
        #                                shuffle=shuffle)
        #
        # elif FLAGS.site == 'site1_test':
        #
        #     filenames = sorted(glob.glob(os.path.join(FLAGS.train_data_path, '*{}*'.format(FLAGS.site))))
        #     filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        #     image, label = read(filename_queue, train)
        #     num_examples = FLAGS.train_size_per_site
        #     print('building queue from :{}....'.format(filenames))
        #     print(image.shape, label.shape)
        #     return generate_batch(example=[image, label], dataset_size=num_examples, batch_size=batch_size,
        #                           shuffle=shuffle)
        if FLAGS.percent == 10:
            filenames = sorted(glob.glob(os.path.join(FLAGS.train_data_path, 'lsun20_balance_site0batch0.tfrecords')))
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
            image, label = read(filename_queue, train)
            num_examples = int(FLAGS.train_size_per_pct * FLAGS.percent*0.1)
            tf.logging.info('labeled inputs: building queue from :{}....'.format(filenames))
            tf.logging.info("image shape:{} , label shape:{}".format(image.shape, label.shape))
            return generate_batch(example=[image, label], dataset_size=num_examples, batch_size=batch_size,
                                  shuffle=shuffle)
        if FLAGS.percent == 20:
            filenames = sorted(glob.glob(os.path.join(FLAGS.train_data_path, 'site0/lsun20_balance_site0batch0.tfrecords')))
            filenames += sorted(glob.glob(os.path.join(FLAGS.train_data_path, 'site0/lsun20_balance_site0batch1.tfrecords')))
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
            example_list = [read(filename_queue, train)
                            for _ in range(len(filenames))]
            num_examples = FLAGS.train_size_per_pct * FLAGS.percent
            tf.logging.info('labeled inputs: building queue from:{}....'.format(filenames))
            tf.logging.info("image shape:{} , label shape:{}".format(example_list[0][0].shape, example_list[0][1].shape))
            return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                       shuffle=shuffle)
    else:
        filenames = sorted(glob.glob(os.path.join(FLAGS.test_data_path, '*batch1*')))
        tf.logging.info(filenames)
        num_examples = FLAGS.test_size
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

        image, label = read(filename_queue, train)
        tf.logging.info("image shape:{} , label shape:{}".format(image.shape, label.shape))
        tf.logging.info('test inputs: building queue from :{}....'.format(filenames))
        return generate_batch(example=[image, label], dataset_size=num_examples, batch_size=batch_size, shuffle=shuffle)

def unlabeled_inputs(batch_size=100,
                     shuffle=True):
    filenames=[]
    if FLAGS.percent == 10:
        filenames += sorted(glob.glob(os.path.join(FLAGS.train_data_path, '*')))
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        example_list = [read(filename_queue, True)
                            for _ in range(len(filenames))]
        tf.logging.info('unlabled inputs: building queue from:{}....'.format(filenames))
        tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
        #assert 0
        num_examples=int(FLAGS.train_size_per_pct * FLAGS.percent)
        return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                   shuffle=shuffle)
    if FLAGS.percent == 20:
        filenames += sorted(glob.glob(os.path.join(FLAGS.train_data_path, '*site0.*')))
        filenames += sorted(glob.glob(os.path.join(FLAGS.train_data_path, '*site1.*')))
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        example_list = [read(filename_queue, True)
                            for _ in range(len(filenames))]
        tf.logging.info('unlabled inputs: building queue from:{}....'.format(filenames))
        tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
        #assert 0
        num_examples=int(FLAGS.train_size_per_pct * FLAGS.percent)
        return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                   shuffle=shuffle)






def main(argv):
    prepare_dataset()


if __name__ == "__main__":
    tf.app.run()
