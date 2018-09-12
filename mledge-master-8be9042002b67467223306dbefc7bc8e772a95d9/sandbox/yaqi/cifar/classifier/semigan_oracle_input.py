

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
tf.app.flags.DEFINE_string('train_data_path', '',
                           'where to store the dataset')


def inputs(batch_size=100,
           train=True,
           shuffle=True, num_epochs=None):

    if train:

        if FLAGS.gantype == 'CTGAN_SEMI':
            filenames = sorted(glob.glob(os.path.join(FLAGS.train_data_path, '*wdtop*')))
            print(filenames)
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
            example_list = [read(filename_queue, train)
                            for _ in range(len(filenames))]
            tf.logging.info('building queue from:{}....'.format(filenames))
            tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
            num_examples = int(FLAGS.num_labeled_examples)
            return generate_batch_join(example=example_list, dataset_size=num_examples, batch_size=batch_size,
                                       shuffle=shuffle)
        else:
            assert 0

    else:
        filenames = ['test_cifar_nozca.tfrecords']
        num_examples = NUM_EXAMPLES_TEST
        filename_queue = generate_filename_queue(filenames, '../Data/', num_epochs)
        image, label = read(filename_queue,train)
        print("labeled_cifar_img_shape:", image.shape, type(image))
        # image = transform(tf.cast(image, tf.float32)) if train else image
        return generate_batch([image, label], num_examples, batch_size, shuffle)



