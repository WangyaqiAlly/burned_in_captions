import os
import numpy as np
import cv2
import tensorflow as tf
import Utils as utils
import logging

num_units_per_layer = 16
model = 'model.proto'
decay_steps = 10000
decay_rate = 0.95


class EdgeNode(object):
    def __init__(self, sess, graph_dir, edge_id, image_type, time, record_path,
                 batch_size, learning_rate,label_rate,cat_dim,con_dim,latent_dim):

        self.logger = logging.getLogger('Central Node')

        self.logger.debug('----------------Edge Node init with ------------- \n graph_dir: {:s}'
                          '\n edge_id: {:s} \n image_type: {:s} \n record_path: {:s} \n batch_size: {:d}'
                          '\n learning_rate: {:f}'
                          .format(graph_dir, edge_id, image_type, record_path, batch_size, learning_rate))

        self.sess = sess
        self.graph_dir = graph_dir
        self.edge_id = edge_id
        self.record_path = record_path
        self._reader = utils.ImageLoader(image_type, graph_dir, edge_id, 1,time)
        self.batch_size = batch_size
        self.initial_learning_rate = learning_rate
        self.label_rate = label_rate
        self.cat_dim = cat_dim
        self.con_dim = con_dim
        self.latent_dim = latent_dim
        #print '++++++++++++++++++++++++++++++++++++++++++'
        #print self.latent_dim
        self.train_dataset = self._reader.train
        self.test_dataset = self._reader.test

        self._build_network()

    def _build_network(self):
        self.logger.info('This is EdgeNode default _build_network() (labelGAN)')
        self.logger.debug('-------------Begin building network --------------------')

        with tf.variable_scope('input_' + str(self.edge_id)):
            self.x = tf.placeholder('float32', [None, 28, 28, 1], name='x')
            self.z = tf.placeholder('float32', [None, self.latent_dim], name='z')
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

        self.saver = tf.train.Saver()

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
    
    def creat_z_cat_nonrand(args, num_sample, digit):
        z_cat = np.zeros((num_sample,args.cat_dim),dtype='float32')
        z_cat[:,digit] = 1
        return z_cat
    def creat_z_cat(self,batch_size,cat_dim):
        z_cat = np.zeros((batch_size,cat_dim),dtype='float32')
        for i in xrange(batch_size):
            ind=np.where(np.random.multinomial(1, np.ones((cat_dim)) / cat_dim)==1)[0][0]
            np.put(z_cat[i],ind,1)
        return z_cat

    def creat_show_z_cat(self,numz):
        z_cat = np.zeros((numz,self.cat_dim),dtype='float32')
        for i in xrange(numz):
            #  ind=np.where(np.random.multinomial(1, np.ones((args.cat_dim)) / args.cat_dim)==1)[0][0]
            ind = i % 10
            np.put(z_cat[i],ind,1)
        return z_cat

    def train(self, total_epoch):

        for i in range(total_epoch):
            # Shuffle data
            data = self.train_dataset.images
            data = data[0:(int(data.shape[0]/self.batch_size)*self.batch_size)]
            label = self.train_dataset.labels
            label = label[0:(int(data.shape[0]/self.batch_size)*self.batch_size)]
            #rng_state = np.random.get_state()
            #np.random.shuffle(self.train_dataset.images)
            #np.random.set_state(rng_state)
            #np.random.shuffle(self.train_dataset.labels)
            #label = np.eye(10)[label]
            rng_state = np.random.get_state()
            np.random.shuffle(data)
            np.random.set_state(rng_state)
            np.random.shuffle(label)
            label = np.eye(10)[label]
            #print label.size
            #print label
            data = data/255.
            data = data-0.5
            print 'data max:',np.max(data)

            ###################### max is 255 need to change 

            # xtest = data[0:1000]
            # ltest = np.argmax(label[0:1000],axis=1)
            
            # print xtest.shape
            # print ltest.shape
            # print 'ltest',ltest
            # utils.visualization_color(xtest,ltest,self.edge_id)

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
                                                       self.phase:True,
                                                       self.z_cat: self.creat_z_cat(self.batch_size,self.cat_dim), 
                                                       self.z_con: np.random.normal(size=(self.batch_size, self.con_dim)),
                                                       self.z_rand: np.random.normal(size=(self.batch_size, self.latent_dim)) })

                et += 1.
                elb += el_
                enb += en_

                _, gl_, gn_ = self.sess.run([self.gtrain, self.gloss, self.gnorm],
                                            feed_dict={self.x: data[j:j + self.batch_size],
                                                       self.l: label[j:j + self.batch_size],
                                                       self.phase:True,
                                                       self.z_cat: self.creat_z_cat(self.batch_size,self.cat_dim), 
                                                       self.z_con: np.random.normal(size=(self.batch_size, self.con_dim)),
                                                       self.z_rand: np.random.normal(size=(self.batch_size, self.latent_dim)) })
                gt += 1.
                glb += gl_
                gnb += gn_

                _, gl_, gn_ = self.sess.run([self.gtrain, self.gloss, self.gnorm],
                                            feed_dict={self.x: data[j:j + self.batch_size],
                                                       self.l: label[j:j + self.batch_size],
                                                       self.phase:True,
                                                       self.z_cat: self.creat_z_cat(self.batch_size,self.cat_dim), 
                                                       self.z_con: np.random.normal(size=(self.batch_size, self.con_dim)),
                                                       self.z_rand: np.random.normal(size=(self.batch_size, self.latent_dim)) })
                gt += 1.
                glb += gl_
                gnb += gn_

            self.logger.info('Edge {:2s}: epoch {:5d}  eloss {:12.8f}  gloss {:12.8f}'
                             'egrad {:12.8f}  ggrad {:12.8f}  learning_rate {:12.8f}'
                             .format(self.edge_id, i, elb / et, glb / gt, enb / et, gnb / gt,
                                     self.sess.run(self.learning_rate)
                                     ))

            x0, l0 = self.sess.run([self.gx, self.gl],
                                   feed_dict={ self.z_cat: self.creat_show_z_cat(10*10), 
                                               self.phase:True, 
                                                       self.z_con: np.random.normal(size=(10*10, self.con_dim)),
                                                       self.z_rand: np.random.normal(size=(10*10, self.latent_dim)) })
            x0 = x0 + 0.5
            x0 = np.clip(x0, 0., 1.) * 255.
            l0 = np.argmax(l0, axis=1)

            utils.visualization_color(x0, l0, self.edge_id)


    def save_model(self, record_path='', edge_id='', time=''):
        if edge_id != '':
            save_path = self.saver.save(self.sess, os.path.join(record_path, edge_id, time, 'model.ckpt'))
        else:
            save_path = self.saver.save(self.sess, record_path)
        self.logger.info("Model saved in file: %s" % save_path)

    def restore_model(self, record_path='', edge_id='', time=''):
        if edge_id != '':
            self.saver.restore(self.sess, os.path.join(record_path, edge_id, time, 'model.ckpt'))
        else:
            self.saver.restore(self.sess, record_path)
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

