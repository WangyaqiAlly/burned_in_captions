import tensorflow as tf
import numpy as np
from retinaBase import Reader
import matplotlib.pyplot as plt
import os
import cv2
import sys

IMAGE_SIZE = 24

class RetinaNode(object):
    def __init__(self, nodepath, record_name, batch_num):
        self._reader = Reader(nodepath, record_name) 
        self._nodename = nodepath
        self._record_name = record_name
        self.batch_num = batch_num
        #self.visualize(self._reader.train.images, self._reader.train.labels)
        self._buildNetwork()
        self._buildDistort()
    '''
    def _buildNetwork(self):
        global_step = tf.Variable(0, trainable=False)
        self.images, self.labels = (self._reader.train_image_batch, self._reader.train_label_batch)
        self.logits = cifar10.inference(self.images)
        self.loss = cifar10.loss(self.logits, self.labels)
        self.train_op = cifar10.train(self.loss, global_step)
    '''
    def _buildDistort(self):
        self._distortX = tf.placeholder(tf.float32, shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
        self._distort_flip = tf.image.random_flip_left_right(self._distortX)
        self._distort_brightness = tf.image.random_brightness(self._distort_flip, max_delta=63)
        self._distort_contrast = tf.image.random_contrast(self._distort_brightness, lower=0.2, upper=1.8)
        self._distort_output = self._distort_contrast

        
    def _buildNetwork(self):
        def _vwwd(shape, stddev, wd):
        # variable with weight decay
            var = tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32))
            if wd is not None:
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss'))
            return var

        def conv2d(name, l_input, w, b):
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1,1,1,1], padding='SAME'), b), name=name)

        def max_pool(name, l_input, ksize, strides):
            return tf.nn.max_pool(l_input, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding='SAME', name=name)

        def norm(name, l_input, lsize=4):
            return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

        def local(name, l_input, w, b):
            return tf.nn.relu(tf.matmul(l_input, w) + b, name=name)

        n_class = 10

        _weights = {
            'wc1': _vwwd([5, 5,  3, 64], stddev=5e-2, wd=0.0),
            'wc2': _vwwd([5, 5, 64, 64], stddev=5e-2, wd=0.0),
            'wl3': _vwwd([IMAGE_SIZE * IMAGE_SIZE * 4, 384],    stddev=0.04, wd=0.004),
            'wl4': _vwwd([384, 192],     stddev=0.04, wd=0.004),
            'out': _vwwd([192, n_class], stddev=1/192.0, wd=0.0),
        }

        _biases = {
            'bc1' :  tf.Variable(tf.constant(value=0.0 ,shape=[64],  dtype=tf.float32)),
            'bc2' :  tf.Variable(tf.constant(value=0.1, shape=[64],  dtype=tf.float32)),
            'bl3' :  tf.Variable(tf.constant(value=0.1, shape=[384], dtype=tf.float32)),
            'bl4' :  tf.Variable(tf.constant(value=0.1, shape=[192], dtype=tf.float32)),
            'out' :  tf.Variable(tf.constant(value=0.0, shape=[n_class],  dtype=tf.float32)),
        }

        self.x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
        self.y_ = tf.placeholder(tf.int64, shape=[None])
        batch_num = tf.Variable(self.batch_num, tf.int64)
        self.keep_prob = tf.placeholder(tf.float32)
        _dropout = self.keep_prob

        conv1 = conv2d('conv1', self.x, _weights['wc1'], _biases['bc1'])
        pool1 = max_pool('pool1', conv1, ksize=3, strides=2)
        norm1 = norm('norm1', pool1, lsize=4)
        print 'norm1', norm1.get_shape()
        norm1 = tf.nn.dropout(norm1, _dropout)

        conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
        # [very interesting, reverse the order]
        norm2 = norm('norm2', conv2, lsize=4)
        pool2 = max_pool('pool2', norm2, ksize=3, strides=2)
        print 'pool2', pool2.get_shape()
        pool2= tf.nn.dropout(pool2, _dropout)
        # [very interesting, delete the dropout]

        pool2 = tf.reshape(pool2, [-1, IMAGE_SIZE * IMAGE_SIZE * 4])
        print 'pool2', pool2.get_shape()
        local3 = local('local3', pool2, _weights['wl3'], _biases['bl3'])

        local4 = local('local4', local3, _weights['wl4'], _biases['bl4'])

        self.softmax = tf.add(tf.matmul(local4, _weights['out']), _biases['out'], name='softmax')

        global_step = tf.Variable(0, trainable=False)
        decay_step = 100
        self.cross_entropy_individual = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.softmax, labels=self.y_)
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.softmax, labels=self.y_))
        '''
        tf.add_to_collection('losses', self.cross_entropy)
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.lr = tf.train.exponential_decay(0.1, global_step, decay_step, 0.1, staircase=True)
        losses = tf.get_collection('losses')
        loss_average = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_average.apply(losses + [self.loss])
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(self.lr)
            grads = opt.compute_gradients(self.loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            self.train_op = tf.no_op(name='train')
        '''
        self.lr = tf.train.exponential_decay(0.001, global_step, decay_step, 0.996, staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy, global_step=global_step)
        self.correct_prediction = tf.equal(tf.argmax(self.softmax, 1), self.y_)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def initialize(self):
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config = config) #tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

        print '%s initialize successfully.' % self._nodename

    def close(self):
        self.sess.close()
        print '%s close successfully.' % self._nodename

    def train(self, total_epoch):
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self.sess, coord=coord)
        accuracy = 0
        loss = 0
        train_accuracy = 0
        i = 0
        last_epoch = 0
        accArr = []
        self._reader.train.start_fill(total_epoch, self.batch_num)
        n = self._reader.train.num_examples
        batch_num_per_epoch = n / self.batch_num 
        while True:
            epoch = i / batch_num_per_epoch + 1 
            if epoch > total_epoch:
                break

            batch = self._reader.train.next_batch_from_queue(self.batch_num)

            if epoch != last_epoch:
                train_accuracy = self.sess.run(self.accuracy, feed_dict={
                    self.x:batch[0], self.y_:batch[1], self.keep_prob:1.0})
                train_loss = self.sess.run(self.cross_entropy, feed_dict={
                    self.x:batch[0], self.y_:batch[1], self.keep_prob:1.0})
                test_accuracy = self.test()
                accArr += [test_accuracy]
                last_epoch = epoch

            sys.stdout.write("\rstep %d, epoch %d, t_acc %8.4lf%% | queue: %d" %
                (i, epoch, test_accuracy*100, self._reader.train._resource_pool.qsize()))
            sys.stdout.flush()

            self.sess.run(self.train_step, feed_dict={
                self.x: batch[0], self.y_:batch[1], self.keep_prob: 0.8})

            i += 1
        
        test_accuracy = self.test()
        print 'test_accuracy:', test_accuracy, '| examples :', self._reader.train.num_examples
        self._reader.train.stop_fill()
        return accArr

    def test(self):
        #print('start testing....')
        batch = [self._reader.test.images, self._reader.test.labels]
        accuracy = self.sess.run(self.accuracy, feed_dict={
                        self.x:batch[0][:,4:28,4:28], 
                        self.y_:batch[1], 
                        self.keep_prob: 1.0})
        return accuracy 

    def save_model(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, os.path.join(self._nodename, 'model.ckpt'))
        print "Model saved in file: %s" % save_path

    def restore_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(self._nodename, 'model.ckpt'))
        print "Model restored."
    
    def updateReader(self):
        self._reader = Reader(self._nodename, self._record_name)
        print 'successfully update reader'

    def visualize(self, images, labels):
        images = np.array(images)
        labels = np.array(labels)
        # Visualize some examples from the dataset.
        # We show a few examples of training images from each class.
        # print images[0]
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        samples_per_class = 7
        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(labels == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=True)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(images[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.show()

