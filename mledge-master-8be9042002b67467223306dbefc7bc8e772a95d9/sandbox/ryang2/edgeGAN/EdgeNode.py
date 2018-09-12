import os
import numpy as np
import cv2
import tensorflow as tf
import Utils as utils
import logging

num_units_per_layer = 16
latent_dimension = 10
model = 'model.proto'
decay_steps = 10000
decay_rate = 0.95


#   data_start_time, data_end_time: Read all the data from time folder start to end
#       e.g. data_start_time = 1, data_end_time = 10 means EdgeNode will read all data in the site
class EdgeNode(object):
    def __init__(self, sess, graph_dir, edge_id, image_type,
                 data_start_time, data_end_time, record_path,
                 batch_size=100, learning_rate=0.0001,
                 save_images=False, image_path="",
                 latent_dimension=10, label_rate=1.0):

        self.logger = logging.getLogger('Edge Node')

        self.logger.debug('----------------Edge Node init with ------------- \n graph_dir: {:s}'
                          '\n edge_id: {:s} \n image_type: {:s} \n record_path: {:s} \n batch_size: {:d}'
                          '\n learning_rate: {:f}'
                          .format(graph_dir, edge_id, image_type, record_path, batch_size, learning_rate))

        self.sess = sess
        self.graph_dir = graph_dir
        self.edge_id = edge_id
        self.image_type = image_type
        self._reader_end_time = data_end_time
        self._reader = utils.ImageLoader(image_type, graph_dir, edge_id, data_start_time, data_end_time)
        self.record_path = record_path
        self.batch_size = batch_size
        self.initial_learning_rate = learning_rate
        self.save_images = save_images
        self.image_path = image_path
        self.latent_dimension = latent_dimension
        self.label_rate = label_rate

        self.train_dataset = self._reader.train
        self.test_dataset = self._reader.test

    def build_network(self):
        self.logger.info('This is EdgeNode default _build_network() (labelGAN)')
        self.logger.debug('-------------Begin building network --------------------')

        with tf.variable_scope('input_' + str(self.edge_id)):
            self.x = tf.placeholder('float32', [None, 28, 28, 1], name='x')
            self.z = tf.placeholder('float32', [None, latent_dimension], name='z')
            self.logger.debug('z: %s' % (self.z,))
            self.l = tf.placeholder('float32', [None, 10], name='l')

        self.ex = self.enet(self.x, self.l)
        self.logger.debug('------------ DISCRIMINATOR NET constructed --------------')

        self.gx, self.gl = self.gnet()
        self.logger.debug('------------- GENERATOR NET constructed -----------------')

        self.egz = self.enet(self.gx, self.gl, True)
        self.logger.debug('------------- egz NET constructed ----------------------')

        self.eloss = -tf.reduce_mean(tf.log(self.ex) + tf.log(1. - self.egz))
        self.gloss = -tf.reduce_mean(tf.log(self.egz))

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, decay_steps,
                                                        decay_rate, staircase=True)

        self.eopt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        self.egrads = self.eopt.compute_gradients(self.eloss,
                                                  var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                             'enet_' + str(self.edge_id)))
        self.etrain = self.eopt.apply_gradients(self.egrads, global_step=self.global_step)
        self.enorm = tf.global_norm([i[0] for i in self.egrads])

        self.gopt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        self.ggrads = self.gopt.compute_gradients(self.gloss,
                                                  var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                             'gnet_' + str(self.edge_id)))
        self.gtrain = self.gopt.apply_gradients(self.ggrads, global_step=self.global_step)
        self.gnorm = tf.global_norm([i[0] for i in self.ggrads])

    def gnet(self, reuse=None):
        self.logger.info("This is EdgeNode default gnet (labelGAN)")
        with tf.variable_scope('gnet_' + str(self.edge_id), reuse=reuse):
            # Rob's labelGAN
            l = tf.layers.dense(inputs=self.z, units=100, activation=tf.nn.elu)
            # print l
            l = tf.layers.dense(inputs=l, units=100, activation=tf.nn.elu)
            # print l
            l = tf.layers.dense(inputs=l, units=100, activation=tf.nn.elu)
            # print l
            l = tf.layers.dense(inputs=l, units=100, activation=tf.nn.elu)
            # print l
            l = tf.layers.dense(inputs=l, units=10, activation=tf.nn.softmax)
            # print l
            l = tf.identity(l, name='lout')
            self.logger.debug('l: %s' % (l,))
            g = tf.layers.dense(inputs=self.z, units=8 * 8 * num_units_per_layer, activation=None)
            # print g
            g = tf.reshape(g, [-1, 8, 8, num_units_per_layer])
            # print g
            g = tf.layers.conv2d(inputs=g, filters=num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.layers.conv2d(inputs=g, filters=num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.image.resize_bilinear(images=g, size=[14, 14])
            # print g
            g = tf.layers.conv2d(inputs=g, filters=num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.layers.conv2d(inputs=g, filters=num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.image.resize_bilinear(images=g, size=[28, 28])
            # print g
            g = tf.layers.conv2d(inputs=g, filters=num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.layers.conv2d(inputs=g, filters=num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.layers.conv2d(inputs=g, filters=1, kernel_size=3,
                                 strides=1, activation=None, padding='same')
            # print g
            g = tf.identity(g, name='gout')
            self.logger.debug('g: %s' % (g,))
        return g, l

    def enet(self, x, l, reuse=None):
        self.logger.info("This is EdgeNode default enet (labelGAN)")
        with tf.variable_scope('enet_' + str(self.edge_id), reuse=reuse):
            e = tf.layers.conv2d(inputs=x, filters=num_units_per_layer, kernel_size=3, strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print e
            e = tf.layers.conv2d(inputs=e, filters=num_units_per_layer, kernel_size=3, strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print e
            e = tf.image.resize_bilinear(images=e, size=[14, 14])
            # print e
            e = tf.layers.conv2d(inputs=e, filters=2 * num_units_per_layer, kernel_size=3, strides=1,
                                 activation=tf.nn.elu, padding='same')
            # print e
            e = tf.layers.conv2d(inputs=e, filters=2 * num_units_per_layer, kernel_size=3, strides=1,
                                 activation=tf.nn.elu, padding='same')
            # print e
            e = tf.image.resize_bilinear(images=e, size=[7, 7]);
            # print e
            e = tf.layers.conv2d(inputs=e, filters=4 * num_units_per_layer, kernel_size=3, strides=1,
                                 activation=tf.nn.elu, padding='same')
            # print e
            e = tf.layers.conv2d(inputs=e, filters=4 * num_units_per_layer, kernel_size=3, strides=1,
                                 activation=tf.nn.elu, padding='same')
            # print e
            e = tf.contrib.layers.flatten(e);
            # print e
            e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu)
            # print e
            e = tf.layers.dense(inputs=e, units=l.shape[1], activation=tf.nn.elu)
            # print e
            e = tf.concat([e, l], axis=1);
            # print e
            e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu)
            # print e
            e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu)
            # print e
            e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu)
            # print e
            e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu)
            # print e
            e = tf.layers.dense(inputs=e, units=10, activation=tf.nn.elu)
            # print e
            e = tf.layers.dense(inputs=e, units=1, activation=tf.sigmoid)
            # print e
            e = tf.identity(e, name='eout')
            self.logger.debug('e: %s' % (e,))
        return e

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.logger.info('EDGE %s is initialized successfully.' % self.edge_id)

    def train(self, total_epoch):

        for i in range(total_epoch):
            # Shuffle data
            data = self.train_dataset.images
            label = self.train_dataset.labels
            self.logger.info("Training data shape: %s" % (data.shape,))
            rng_state = np.random.get_state()
            np.random.shuffle(self.train_dataset.images)
            np.random.set_state(rng_state)
            np.random.shuffle(self.train_dataset.labels)
            label = np.eye(10)[label]

            elb = 0.
            glb = 0.
            enb = 0.
            gnb = 0.
            et = 0.
            gt = 0.
            for j in range(0, data.shape[0], self.batch_size):
                _, el_, en_ = self.sess.run([self.etrain, self.eloss, self.enorm],
                                            feed_dict={self.x: data[j:j + self.batch_size],
                                                       self.l: label[j:j + self.batch_size],
                                                       self.z: np.random.randn(self.batch_size, latent_dimension)})
                et += 1.
                elb += el_
                enb += en_

                _, gl_, gn_ = self.sess.run([self.gtrain, self.gloss, self.gnorm],
                                            feed_dict={self.z: np.random.randn(self.batch_size, latent_dimension)})
                gt += 1.
                glb += gl_
                gnb += gn_

                _, gl_, gn_ = self.sess.run([self.gtrain, self.gloss, self.gnorm],
                                            feed_dict={self.z: np.random.randn(self.batch_size, latent_dimension)})
                gt += 1.
                glb += gl_
                gnb += gn_

                utils.update_progress(i, int((j + 1.0) / (data.shape[0] * 1.0) * 1000 + 1) / 1000.0)

            utils.update_progress(i, 1)

            self.logger.info('Edge {:2s}: epoch {:5d}  eloss {:7.3f}  gloss {:7.3f}'
                             'egrad {:7.3f}  ggrad {:7.3f}  learning_rate {:7.3f}'
                             .format(self.edge_id, i, elb / et, glb / gt, enb / et, gnb / gt,
                                     self.sess.run(self.learning_rate)
                                     ))

            x0, l0 = self.sess.run([self.gx, self.gl],
                                   feed_dict={self.z: np.random.randn(self.batch_size * 5, latent_dimension)})
            x0 = (np.clip(x0, -0.5, 0.5) + 0.5) * 255.
            l0 = np.argmax(l0, axis=1)

            utils.visualization_grey(x0, l0, self.edge_id, self.save_images, self.image_path)

    def update_data(self, data_start_time, data_end_time, image_path):
        if self._reader_end_time != data_end_time:
            self._reader = utils.ImageLoader(self.image_type, self.graph_dir, self.edge_id,
                                             data_start_time, data_end_time)
            self.train_dataset = self._reader.train
            self.test_dataset = self._reader.test
            self.image_path = image_path

    def save_model(self, record_path='', edge_id='', time=''):
        saver = tf.train.Saver()
        if edge_id != '':
            save_path = saver.save(self.sess, os.path.join(record_path, edge_id, time, 'model.ckpt'))
        else:
            save_path = saver.save(self.sess, record_path)
        self.logger.info("Model saved in file: %s" % save_path)

    def restore_model(self, record_path='', edge_id='', time=''):
        saver = tf.train.Saver()
        if edge_id != '':
            saver.restore(self.sess, os.path.join(record_path, edge_id, time, 'model.ckpt'))
        else:
            saver.restore(self.sess, record_path)
        self.logger.info("Model restored for edge %s" % str(self.edge_id))

    # Used for debugging
    def print_vars(self):
        print '======================================'
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print v.name

        print '=========================================\n'
        all_ops = self.sess.graph.get_operations()
        for op in all_ops:
            print op.name, op.values()

        print '=========================================\n'

    def test(self):
        raise NotImplemented
