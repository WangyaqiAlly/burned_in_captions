import tensorflow as tf
import numpy as np
from retinaBase import Reader
import matplotlib.pyplot as plt
import os

IMAGE_SIZE = 28

class RetinaNode(object):
    def __init__(self, nodepath, record_name, batch_num):
        self._reader = Reader(nodepath, record_name) 
        self._nodename = nodepath
        self._record_name = record_name
        self.batch_num = batch_num
        #self.visualize(self._reader.train.images, self._reader.train.labels)
        self._buildNetwork()

    def _buildNetwork(self):
	def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv_2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1,2,2,1],
                                    strides=[1,2,2,1], padding='SAME')

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        x_image = tf.reshape(self.x, [-1,28,28,1])
        self.y_ = tf.placeholder(tf.int64, shape=[None])

        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.softmax = tf.nn.softmax(y_conv)

        self.cross_entropy_individual = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv, labels=self.y_)
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv, labels=self.y_))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.prediction = tf.argmax(y_conv, 1)
        self.correct_prediction = tf.equal(tf.argmax(y_conv, 1), self.y_)
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
        trainlossArr = []
        trainaccArr = []
        testlossArr = []
        testaccArr = []
        last_epoch = 0
        while True:
            batch = self._reader.train.next_batch(self.batch_num, distort=False)

            epoch = self._reader.train.epochs_completed + 1
            if epoch > total_epoch:
                break

            if epoch != last_epoch:
                train_accuracy = self.sess.run(self.accuracy, feed_dict={
                    self.x:batch[0], self.y_:batch[1], self.keep_prob:1.0})
                trainaccArr += [train_accuracy]
                train_loss = self.sess.run(self.cross_entropy, feed_dict={
                    self.x:batch[0], self.y_:batch[1], self.keep_prob:1.0})
                trainlossArr += [train_loss]
                test_accuracy, test_loss = self.test()
                testlossArr += [test_loss]
                testaccArr += [test_accuracy]
                print("epoch %d | train accuracy %12.8lf, train loss %12.8lf | test accuracy %12.8lf, test loss %12.8lf" %
                    (epoch, train_accuracy, train_loss, test_accuracy, test_loss))
                last_epoch = epoch

            self.sess.run(self.train_step, feed_dict={
                self.x: batch[0], self.y_:batch[1], self.keep_prob: 0.8})

        
        return trainlossArr, trainaccArr, testlossArr, testaccArr 

    def test(self):
        batch = [self._reader.test.images, self._reader.test.labels]
        loss = self.sess.run(self.cross_entropy, feed_dict={
                        self.x:batch[0], 
                        self.y_:batch[1], 
                        self.keep_prob: 1.0})
        accuracy = self.sess.run(self.accuracy, feed_dict={
                        self.x:batch[0], 
                        self.y_:batch[1], 
                        self.keep_prob: 1.0})
        return accuracy, loss

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

    def visualize(self, images, labels):
        # Visualize some examples from the dataset.
        # We show a few examples of training images from each class.
        # print images[0]
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        num_classes = len(classes)
        samples_per_class = 7
        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(labels == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow((255*images[idx]).astype('uint8').reshape(28,28))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.show()

