import tensorflow as tf
import Utils as utils
import os
import numpy as np
import logging

num_units_per_layer = 16
latent_dimension = 10


class CentralNode(object):
    def __init__(self,
                 graph_dir,
                 graph_type='mnist',
                 batch_size=200,
                 use_fake_images=True,
                 generator_record_prefix=None,
                 num_generated_images=100,
                 edge_ids=None,
                 record_time=1,
                 learning_rate=0.0001
                 ):

        if edge_ids is None:
            edge_ids = ['1', '2', '3']

        self.sess = tf.Session()

        self.graph_dir = graph_dir
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.use_fake_images = use_fake_images
        self.generator_record_prefix = generator_record_prefix
        self.num_generated_images = num_generated_images
        self.edge_ids = edge_ids
        self.learning_rate = learning_rate

        if use_fake_images:
            record_dir_sets = []
            for edge_id in edge_ids:
                record_dir_sets.append(
                    os.path.join(generator_record_prefix, str(edge_id), 'time_' + str(record_time), 'model.ckpt'))
            self._reader = utils.ImageGenerator(self.num_generated_images, graph_type, graph_dir, record_dir_sets,
                                                edge_ids)
        else:
            # TODO: need to be modified
            self._reader = utils.ImageLoader(graph_type, graph_dir)

        self.train_dataset = self._reader.train
        self.test_dataset = self._reader.test

        self.logger = logging.getLogger('Central Node')

        self._build_network()

    def _build_network(self):
        with tf.variable_scope('input_center'):
            self.x = tf.placeholder('float32', [None, 32, 32, 3], name='x')
            self.logger.debug('x: %s' % (self.x,))
            self.y = tf.placeholder('int32', [None], name='y')
            self.logger.debug('y: %s' % (self.y,))
            #self._cnett = self.classifier(0.5,phase=True)
            

        with tf.variable_scope('loss_center'):
            self._cnet = self.classifier(0.5,phase=True,reuse=None)
            self.eloss = tf.losses.sparse_softmax_cross_entropy(self.y, self._cnet)
            self.eopt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.egrads = self.eopt.compute_gradients(self.eloss)
            self.etrain = self.eopt.apply_gradients(self.egrads)
            self.enorm = tf.global_norm([i[0] for i in self.egrads])
            self.epred = tf.nn.softmax(self._cnet)
            self._cnetreal = self.classifier(1.0,phase=True,reuse=True)
            self.epred2 = tf.nn.softmax(self._cnetreal)
            
            

        self.logger.info('\n\n ===== Center Network built successfully ====')

    def classifier(self,keep_rate,phase=True,reuse=None):
        with tf.variable_scope("classifier",reuse=reuse):
            e = tf.layers.conv2d(inputs=self.x, filters=16, kernel_size=3, strides=2,activation=None, padding='same') ;
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
            e = tf.nn.dropout(e,keep_rate)
            e = tf.layers.conv2d(inputs=e, filters=32, kernel_size=3, strides=1,activation=None, padding='same') ; 
            e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn')
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
        
            e = tf.nn.dropout(e,keep_rate)
        
            #e = tf.image.resize_bilinear(images=e,size=[16,16]) ; print e
            e = tf.layers.conv2d(inputs=e, filters=64, kernel_size=3, strides=2,activation=None, padding='same') ; 
            e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn2')
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
            e = tf.nn.dropout(e,keep_rate)
        
            e = tf.layers.conv2d(inputs=e, filters=128, kernel_size=3, strides=1,activation=None, padding='same') ; 
            e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn3')
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
            e = tf.nn.dropout(e,keep_rate)
        
            #e = tf.image.resize_bilinear(images=e,size=[8,8]) ; print e
            e = tf.layers.conv2d(inputs=e, filters=256, kernel_size=3, strides=2,activation=None, padding='same'); 
            e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn4')
            #e = self.leaky_relu(e)
            e = tf.where(tf.greater(e, 0), e, 0.2 * e)
            e = tf.nn.dropout(e,keep_rate)
        
            e = tf.layers.conv2d(inputs=e, filters=512, kernel_size=3, strides=1,activation=None, padding='same') ; 
            e = tf.contrib.layers.flatten(e) ; print e
        
            cat = tf.layers.dense(inputs=e,units=10, activation=None); 
         

        self.logger.info('---- Classifier net built ----')
        return cat

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.logger.info('CENTER is initialized successfully.')

    def train(self, total_epoch=100):
        data = self.train_dataset.images
        label = self.train_dataset.labels
        data_test = self.test_dataset.images
        label_test = self.test_dataset.labels
        for i in range(total_epoch):
            rng_state = np.random.get_state()
            np.random.shuffle(self.train_dataset.images)
            np.random.set_state(rng_state)
            np.random.shuffle(self.train_dataset.labels)

            self.logger.debug('data.shape', data.shape, 'label.shape', label.shape, 'test_data', data_test.shape,
                              'test_label', label_test.shape)

            # train
            el = 0.
            en = 0.
            t = 0.
            for j in range(0, data.shape[0], self.batch_size):
                _, el_, en_ = self.sess.run([self.etrain, self.eloss, self.enorm],
                                            feed_dict={self.x: data[j:j + self.batch_size],
                                                       self.y: label[j:j + self.batch_size]})
                el += el_
                en += en_
                t += 1.

            # test
            acc = 0.
            tt = 0.
            for j in range(0, data_test.shape[0], self.batch_size):
                p = self.sess.run(self.epred2, feed_dict={self.x: data_test[j:j + self.batch_size]})
                acc += np.mean(np.argmax(p, axis=1) == label_test[j:j + self.batch_size])
                tt += 1.

            self.logger.info('epoch {:3d} eloss {:f} enorm {:f} accuracy {:f}'.format(i, el / t, en / t, acc / tt))

    #
    # ===================Rarely use==================
    #
    def save_model(self, record_path, edge_id='', time=''):
        saver = tf.train.Saver()
        if edge_id != '':
            save_path = saver.save(self.sess, os.path.join(record_path, edge_id, time, 'model.ckpt'))
        else:
            save_path = saver.save(self.sess, record_path)
        self.logger.info("Model saved in file: %s" % save_path)

    def restore_model(self, record_path, edge_id='', time=''):
        saver = tf.train.Saver()
        save_path = os.path.join(record_path, edge_id, time, 'model.ckpt')
        saver.restore(self.sess, save_path)
        self.logger.info("Center model restored")

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
