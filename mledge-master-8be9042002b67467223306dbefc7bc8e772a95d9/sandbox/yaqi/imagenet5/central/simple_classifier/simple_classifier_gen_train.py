import tensorflow as tf
import numpy as np
import os 
import cv2
import sys

tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]
    tag  = tag + '_100'

if not os.path.exists(tag):
    os.makedirs(tag)
    os.makedirs(tag + '/result')

batch_size = 256
IMAGE_SIZE = 32
num_epoches = 1000


#### Read training data
train_data = np.load('../ALI/uncon_gen/images.npy')
train_labels = np.load('../ALI/uncon_gen/labels.npy')

#### Read validation data
vali_data = np.load('./data/val/images.npy')
vali_labels = np.load('./data/val/labels.npy')
ind = np.arange(vali_data.shape[0])
np.random.shuffle(ind)
vali_data = vali_data[ind]
vali_labels = vali_labels[ind]
# vali_data = vali_data[:1280]
# vali_labels = vali_labels[:1280]


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
        
n_class = 5

_weights = {
    'wc1': _vwwd([5, 5,  3, 64], stddev=5e-2, wd=0.0),
    'wc2': _vwwd([5, 5, 64, 64], stddev=5e-2, wd=0.0),
    'wl3': _vwwd([IMAGE_SIZE * IMAGE_SIZE * 4, 384],    stddev=0.04, wd=0.004),
    'wl4': _vwwd([384, 192],     stddev=0.04, wd=0.004),
    'out': _vwwd([192, n_class], stddev=1/192.0, wd=0.0),
}
_biases = {
    'bc1' : tf.Variable(tf.constant(value=0.0 ,shape=[64],  dtype=tf.float32)),
    'bc2' : tf.Variable(tf.constant(value=0.1, shape=[64],  dtype=tf.float32)),
    'bl3' : tf.Variable(tf.constant(value=0.1, shape=[384], dtype=tf.float32)),
    'bl4' : tf.Variable(tf.constant(value=0.1, shape=[192], dtype=tf.float32)),
    'out' : tf.Variable(tf.constant(value=0.0, shape=[n_class],  dtype=tf.float32)),
}
        

x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.int64, shape=[None])
batch_num = tf.Variable(batch_size, tf.int64)
keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
        
conv1 = conv2d('conv1', x, _weights['wc1'], _biases['bc1'])
pool1 = max_pool('pool1', conv1, ksize=3, strides=2)
norm1 = norm('norm1', pool1, lsize=4)
print 'norm1', norm1.get_shape()
norm1 = tf.nn.dropout(norm1, keep_prob)
conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
# [very interesting, reverse the order]
norm2 = norm('norm2', conv2, lsize=4)
pool2 = max_pool('pool2', norm2, ksize=3, strides=2)
print 'pool2', pool2.get_shape()
pool2= tf.nn.dropout(pool2, keep_prob)
# [very interesting, delete the dropout]
pool2 = tf.reshape(pool2, [-1, IMAGE_SIZE * IMAGE_SIZE * 4])
print 'pool2', pool2.get_shape()
local3 = local('local3', pool2, _weights['wl3'], _biases['bl3'])
local4 = local('local4', local3, _weights['wl4'], _biases['bl4'])

softmax = tf.add(tf.matmul(local4, _weights['out']), _biases['out'], name='softmax')
cross_entropy_individual = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax, labels=y)
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax, labels=y))

opt = tf.train.AdamOptimizer(0.001)
train_step = opt.minimize(cross_entropy)
grad = opt.compute_gradients(cross_entropy)
norm = tf.global_norm([i[0] for i in grad])
prediction = tf.argmax(softmax, 1)
correct_prediction = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

f_training = open(tag + '/training_record.txt', 'w')
f_testing = open(tag + '/testing_record.txt', 'w')
f_prediction = open(tag + '/preditions.txt', 'w')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # model_path = 'model'
    # saver.restore(sess, model_path)
    for epoch in xrange(num_epoches):

        if epoch % 10 == 0:
            acc = 0.0
            counter = 0
            for batch in range(0, vali_data.shape[0]-batch_size, batch_size):
                counter += 1
                input_x = vali_data[batch:batch+batch_size]
                input_x = (input_x / 255.0) - 0.5
                input_y = vali_labels[batch:batch+batch_size] 
                acc += sess.run(accuracy, feed_dict = {x: input_x, y: input_y, keep_prob:1.0})
            acc /= counter
            print '#############################################'
            print 'Epoch:', str(epoch), 'test_acc:', str(acc)
            f_testing.write('Epoch: ' + str(epoch) + ' test_acc: ' +  str(acc) + '\n')

        ind = np.arange(train_data.shape[0])
        np.random.shuffle(ind)
        train_data = train_data[ind]
        train_labels = train_labels[ind]
        acc = 0.0
        counter = 0
        for batch in range(0, train_data.shape[0]-batch_size, batch_size):
            counter += 1
            input_x = train_data[batch:batch+batch_size]
            input_x = np.clip(input_x,0.,1.)*255.- 0.5
            input_y = train_labels[batch:batch+batch_size]
            sess.run(train_step, feed_dict = {x: input_x, y: input_y, keep_prob:0.8})
            if batch == 0:
                norm_, train_loss_, batch_acc = sess.run([norm, cross_entropy, accuracy], feed_dict = {x: input_x, y: input_y, keep_prob:1.0})
                print 'Epoch:', str(epoch), 'train_loss:', str(train_loss_), 'norm:', str(norm_), 'batch_acc:', str(batch_acc)
                f_training.write('Epoch: ' + str(epoch) + ' train_loss: ' +  str(train_loss_) +  ' norm: ' + str(norm_) + ' batch_acc: ' + str(batch_acc)+ '\n')
    savepath = saver.save(sess, tag +'/model'+str(epoch))
    print 'saving ',savepath


    ####  For confusion matrix
    counter = 0
    for batch in range(0, vali_data.shape[0]-batch_size, batch_size):
        counter += 1
        input_x_ori = vali_data[batch:batch+batch_size]
        input_x = (input_x_ori / 255.0) - 0.5
        input_y = vali_labels[batch:batch+batch_size] 
        lab = sess.run(prediction, feed_dict = {x: input_x, y: input_y, keep_prob:1.0})
        for i in range(batch_size):
            #cv2.imwrite(tag +'/result/' + str(counter) + '_' + str(input_y[i])+ '_' + str(lab[i]) + '.bmp', input_x_ori[i])
            counter += 1
            f_prediction.write(str(input_y[i])+ ' ' + str(lab[i]) + '\n')

f_training.close()
f_testing.close()
f_prediction.close()

