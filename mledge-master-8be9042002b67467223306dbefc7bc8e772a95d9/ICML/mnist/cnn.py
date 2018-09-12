import tensorflow as tf
import numpy
import sys, os
import layers as L

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('keep_prob_hidden', 0.5, "dropout rate")
tf.app.flags.DEFINE_float('lrelu_a', 0.1, "lrelu slope")
tf.app.flags.DEFINE_boolean('top_bn', False, "")


def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):

    h = x
    rng = numpy.random.RandomState(seed)
    print h


    h = L.conv(h, ksize=5, stride=1, f_in=1, f_out=32, padding='VALID', seed=rng.randint(123456), name='c1')
    h = L.max_pool(h, ksize=2, stride=2, padding='VALID')
    # h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b1'), FLAGS.lrelu_a)
    h = L.lrelu(h, FLAGS.lrelu_a)
    print h

    h = L.conv(h, ksize=5, stride=1, f_in=32, f_out=64, padding='VALID',seed=rng.randint(123456), name='c2')
    h = tf.nn.dropout(h, keep_prob=FLAGS.keep_prob_hidden, seed=rng.randint(123456)) if stochastic else h
    h = L.max_pool(h, ksize=2, stride=2, padding='VALID')
#    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b2'), FLAGS.lrelu_a)
    h = L.lrelu(h, FLAGS.lrelu_a)
    print h

#    h = tf.reduce_mean(h, reduction_indices=[1, 2])  # Global average pooling
    h = tf.layers.flatten(h)
    print h
    
    h = L.fc(h, 64*4*4, 512, seed=rng.randint(123456), name='fc1')
    h = L.lrelu(h, FLAGS.lrelu_a)
    print h

    h = tf.nn.dropout(h, keep_prob=FLAGS.keep_prob_hidden, seed=rng.randint(123456)) if stochastic else h
    h = L.fc(h, 512,  10, seed=rng.randint(123456), name='fc2')
    print h

#    if FLAGS.top_bn:
#        h = L.bn(h, 10, is_training=is_training,
#                 update_batch_stats=update_batch_stats, name='bfc')

    return h
