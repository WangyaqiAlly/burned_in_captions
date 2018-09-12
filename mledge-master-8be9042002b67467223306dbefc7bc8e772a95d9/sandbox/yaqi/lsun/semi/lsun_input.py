
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

tf.app.flags.DEFINE_string('labeled_data_path', '../Data/lsun20-tf-balance/',
                           'where to load the labeled dataset')

tf.app.flags.DEFINE_string('unlabeled_data_path', ['../Data/generated_data/10percent_site0_model_44999/',
                                                   '../Data/generated_data/10percent_site2_model_149999/',
                                                   '../Data/generated_data/10percent_site4_model_424999',
                                                   '../Data/generated_data/10percent_site5_model_149999'],
                           'where to load the unlabeled dataset')
tf.app.flags.DEFINE_string('test_data_path', '../Data/lsun20-tf-balance/test/',
                           'where to store the dataset')


def inputs(batch_size=100,
           train=True,
           shuffle=True, num_epochs=None):
    if train:
        # if FLAGS.site_num<1:
        #     filenames = []
        #     filenames += sorted(
        #             glob.glob(os.path.join(FLAGS.labeled_data_path, 'lsun20_balance.tfrecords')))
        #     filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        #     example_list = [read(filename_queue, train)
        #                     for _ in range(len(filenames))]
        #     num_examples = int(FLAGS.train_size_per_site * FLAGS.site_num)
        #     print('building queue from:{}....'.format(filenames))
        #     print(example_list[0][0].shape, example_list[0][1].shape)
        #     return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
        #                                shuffle=shuffle)

        filenames = []
        for site_i in FLAGS.sites:
            filenames += sorted(
                glob.glob(os.path.join(FLAGS.labeled_data_path, 'site{}/lsun20_balance_site{}batch0.tfrecords'.format(site_i,site_i))))
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        example_list = [read(filename_queue, train,False)
                        for _ in range(len(filenames))]
        num_examples = FLAGS.train_size_per_pct * FLAGS.site_num
        print('building labeled queue from:{}....'.format(filenames))
        print(example_list[0][0].shape, example_list[0][1].shape)
        return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                   shuffle=shuffle)

    else:
        filenames = sorted(glob.glob(os.path.join(FLAGS.test_data_path, 'lsun20_balance_site0_testbatch1.tfrecords')))
        #filenames = sorted(glob.glob(os.path.join(FLAGS.test_data_path, 'lsun20_balance_test.tfrecords')))
        print(filenames)
        num_examples = FLAGS.test_size
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

        image, label = read(filename_queue, train,False)
        print(image.shape, label.shape)
        print('building test queue from :{}....'.format(filenames))
        return generate_batch(example=[image, label], dataset_size=num_examples, batch_size=FLAGS.eval_batch_size, shuffle=shuffle)






def unlabeled_inputs(batch_size=100,
                     shuffle=True):

    if FLAGS.unlabeled_type == 'real':
        filenames=[]
        for i in xrange(10):
            filenames += sorted(
                glob.glob(os.path.join(FLAGS.labeled_data_path, 'lsun20_balance_site{}.tfrecords'.format(i))))
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        example_list = [read(filename_queue, True,False)
                            for _ in range(len(filenames))]
        tf.logging.info('building queue from:{}....'.format(filenames))
        tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
        num_examples=FLAGS.train_size_per_site*len(filenames)
        return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                   shuffle=shuffle)
    if FLAGS.unlabeled_type == 'CTGAN_SEMI':
        filenames=[]
        for site_i in FLAGS.sites:
            filenames += sorted(
                glob.glob(os.path.join(FLAGS.labeled_data_path, 'site{}/lsun20_balance_site{}batch0.tfrecords'.format(site_i,site_i))))
        for train_path in FLAGS.unlabeled_data_path:
            filenames+= sorted(
                glob.glob(os.path.join(train_path, '*')))

        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        example_list = [read(filename_queue, True,True)
                            for _ in range(len(filenames))]
        tf.logging.info('building  unlabeled queue from:{}....'.format(filenames))
        tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
        num_examples=FLAGS.train_size_per_pct*len(filenames)
        return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                   shuffle=shuffle)


