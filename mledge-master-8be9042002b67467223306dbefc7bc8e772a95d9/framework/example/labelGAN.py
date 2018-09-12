import tensorflow as tf
import numpy as np
from EdgeNode import EdgeNode
import Utils as utils


class labelGAN(EdgeNode):
    def __init__(self, sess, graph_dir, edge_id, image_type, data_start_time, data_end_time,
                 record_path, batch_size, learning_rate, num_units_per_layer,
                 save_images=False, image_path=''):
        EdgeNode.__init__(self, sess, graph_dir, edge_id,
                          image_type, data_start_time, data_end_time,
                          record_path, batch_size, learning_rate,
                          save_images, image_path)

        self.num_units_per_layer = num_units_per_layer
        self.decay_steps = 10000
        self.decay_rate = 0.95

    def build_network(self):
        self.logger.debug('-------------Begin building network --------------------')

        with tf.variable_scope('input_' + str(self.edge_id)):
            self.x = tf.placeholder('float32', [None, 28, 28, 1], name='x')
            self.z = tf.placeholder('float32', [None, self.latent_dimension], name='z')
            self.logger.debug('z: %s' % (self.z,))
            self.l = tf.placeholder('float32', [None, 10], name='l')

        ex = self.enet(self.x, self.l)
        self.logger.debug('------------ DISCRIMINATOR NET constructed --------------')

        self.gx, self.gl = self.gnet()
        self.logger.debug('------------- GENERATOR NET constructed -----------------')

        egz = self.enet(self.gx, self.gl, True)
        self.logger.debug('------------- egz NET constructed ----------------------')

        self.eloss = -tf.reduce_mean(tf.log(ex) + tf.log(1. - egz))
        self.gloss = -tf.reduce_mean(tf.log(egz))

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step,
                                                        self.decay_steps, self.decay_rate, staircase=True)

        eopt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        egrads = eopt.compute_gradients(self.eloss,
                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                   'enet_' + str(self.edge_id)))
        self.etrain = eopt.apply_gradients(egrads, global_step=global_step)
        self.enorm = tf.global_norm([i[0] for i in egrads])

        gopt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        ggrads = gopt.compute_gradients(self.gloss,
                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                   'gnet_' + str(self.edge_id)))
        self.gtrain = gopt.apply_gradients(ggrads, global_step=global_step)
        self.gnorm = tf.global_norm([i[0] for i in ggrads])

    def gnet(self, reuse=None):
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
            g = tf.layers.dense(inputs=self.z, units=8 * 8 * self.num_units_per_layer, activation=None)
            # print g
            g = tf.reshape(g, [-1, 8, 8, self.num_units_per_layer])
            # print g
            g = tf.layers.conv2d(inputs=g, filters=self.num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.layers.conv2d(inputs=g, filters=self.num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.image.resize_bilinear(images=g, size=[14, 14])
            # print g
            g = tf.layers.conv2d(inputs=g, filters=self.num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.layers.conv2d(inputs=g, filters=self.num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.image.resize_bilinear(images=g, size=[28, 28])
            # print g
            g = tf.layers.conv2d(inputs=g, filters=self.num_units_per_layer, kernel_size=3,
                                 strides=1, activation=tf.nn.elu,
                                 padding='same')
            # print g
            g = tf.layers.conv2d(inputs=g, filters=self.num_units_per_layer, kernel_size=3,
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
        with tf.variable_scope('enet_' + str(self.edge_id), reuse=reuse):
            e = tf.layers.conv2d(inputs=x, filters=self.num_units_per_layer, kernel_size=3, strides=1,
                                 activation=tf.nn.elu,
                                 padding='same')
            # print e
            e = tf.layers.conv2d(inputs=e, filters=self.num_units_per_layer, kernel_size=3, strides=1,
                                 activation=tf.nn.elu,
                                 padding='same')
            # print e
            e = tf.image.resize_bilinear(images=e, size=[14, 14])
            # print e
            e = tf.layers.conv2d(inputs=e, filters=2 * self.num_units_per_layer, kernel_size=3, strides=1,
                                 activation=tf.nn.elu, padding='same')
            # print e
            e = tf.layers.conv2d(inputs=e, filters=2 * self.num_units_per_layer, kernel_size=3, strides=1,
                                 activation=tf.nn.elu, padding='same')
            # print e
            e = tf.image.resize_bilinear(images=e, size=[7, 7]);
            # print e
            e = tf.layers.conv2d(inputs=e, filters=4 * self.num_units_per_layer, kernel_size=3, strides=1,
                                 activation=tf.nn.elu, padding='same')
            # print e
            e = tf.layers.conv2d(inputs=e, filters=4 * self.num_units_per_layer, kernel_size=3, strides=1,
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
                                                       self.z: np.random.randn(self.batch_size, self.latent_dimension)})
                et += 1.
                elb += el_
                enb += en_

                _, gl_, gn_ = self.sess.run([self.gtrain, self.gloss, self.gnorm],
                                            feed_dict={self.z: np.random.randn(self.batch_size, self.latent_dimension)})
                gt += 1.
                glb += gl_
                gnb += gn_

                _, gl_, gn_ = self.sess.run([self.gtrain, self.gloss, self.gnorm],
                                            feed_dict={self.z: np.random.randn(self.batch_size, self.latent_dimension)})
                gt += 1.
                glb += gl_
                gnb += gn_

                utils.update_progress(i, int((j + 1.0) / (data.shape[0] * 1.0) * 1000 + 1) / 1000.0)

            utils.update_progress(i, 1)

            self.logger.info('Edge {:2s}: epoch {:3d} eloss {:7.3f} gloss {:7.3f} '
                             'egrad {:7.3f} ggrad {:7.3f} learning_rate {:5.3f}'
                             .format(self.edge_id, i, elb / et, glb / gt, enb / et, gnb / gt,
                                     self.sess.run(self.learning_rate)
                                     ))

            x0, l0 = self.sess.run([self.gx, self.gl],
                                   feed_dict={self.z: np.random.randn(self.batch_size * 5, self.latent_dimension)})
            x0 = (np.clip(x0, -0.5, 0.5) + 0.5) * 255.
            l0 = np.argmax(l0, axis=1)

            utils.visualization_grey(x0, l0, self.edge_id, True, self.image_path)
