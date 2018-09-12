'''
    classify.py:  train and test simply classifier over tiny-images (32x32) 
'''

import tensorflow as tf
import numpy as np
import os
import cv2

batch_size = 128
IMAGE_SIZE = 32
num_epoches = 100

classlist = []
classlist.append('invertebrate')
classlist.append('bird')
classlist.append('vehicle')
classlist.append('dog')
classlist.append('clothing')

def load_dataset(basedir, classlist): 

    nimglist = []
    for i, cname in enumerate(classlist): 
 	classdir = os.path.join(basedir, '%02d-%s' % (i, cname))
 	imglist = os.listdir(classdir)
	nimglist.append(len(imglist))

    print 'images per class: ', nimglist
 
    nimg = sum(nimglist)
    d = np.zeros((nimg,32, 32, 3))
    l = np.zeros(nimg)
    idx = 0
    for i, cname in enumerate(classlist): 
 	classdir = os.path.join(basedir, '%02d-%s' % (i, cname))
 	imglist = os.listdir(classdir)
	for imgfile in imglist:
	    bmpfile = os.path.join(classdir, imgfile) 	    
	    img = cv2.imread(bmpfile)
	    print bmpfile, ' | ', img.shape
 	    
	    d[idx,:,:,:]=img
	    l[idx]=i
	    idx = idx + 1

    return d, l

def rshuffle(d, l): 

    'randomly shuffle dataset'
    ind = np.arange(d.shape[0])
    np.random.shuffle(ind)
    d = d[ind]
    l = l[ind]
    return d, l

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

def load_weights(n_class): 

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

    return _weights, _biases

# --------- start of main procedures -----------
'load data'

# datadir = '/home/xiaoqzhu/dataset/imagenet-plus-5-class'
datadir = '/home/xiaoqzhu/dataset/imagenet-plus-5-class-jpg'
traindir = os.path.join(datadir, 'train')
validdir = os.path.join(datadir, 'validation')

train_data, train_labels = load_dataset(traindir, classlist)
valid_data, valid_labels = load_dataset(validdir, classlist)

print 'train_data.shape: ', train_data.shape, ' train_labels.shape: ', train_labels.shape
print 'valid_data.shape: ', valid_data.shape, ' valid_labels.shape: ', valid_labels.shape

'set up NN in tf'
nclass = len(classlist)

x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.int64, shape=[None])
batch_num = tf.Variable(batch_size, tf.int64)
keep_prob = tf.placeholder(dtype=tf.float32, shape=[])

_weights, _biases = load_weights(nclass)
conv1 = conv2d('conv1', x, _weights['wc1'], _biases['bc1'])
pool1 = max_pool('pool1', conv1, ksize=3, strides=2)
norm1 = norm('norm1', pool1, lsize=4)
print 'norm1', norm1.get_shape()
norm1 = tf.nn.dropout(norm1, keep_prob)
conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
norm2 = norm('norm2', conv2, lsize=4)
pool2 = max_pool('pool2', norm2, ksize=3, strides=2)
print 'pool2', pool2.get_shape()
pool2= tf.nn.dropout(pool2, keep_prob)
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
correct_prediction = tf.equal(tf.argmax(softmax, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'start training & testing session in tf '
saver = tf.train.Saver()
with open('training_record.txt', 'w') as f:

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # model_path = '.......'
        # saver.restore(sess, model_path)
        for epoch in xrange(num_epoches):

	    'test once every 10 epochs'
            if epoch % 10 == 0:
		'randomly shuffle data'
		dv, lv = rshuffle(valid_data, valid_labels)

#            	ind = np.arange(valid_data.shape[0])
#            	np.random.shuffle(ind)
#	        vali_data = valid_data[ind]
#                vali_labels = valid_labels[ind]
                acc = 0.0
                counter = 0
                for batch in range(0, dv.shape[0], batch_size):
                    counter += 1
                    xv = dv[batch:batch+batch_size]
                    xv = (xv / 255.0) - 0.5
                    yv = lv[batch:batch+batch_size]
                    acc += sess.run(accuracy, feed_dict = {x: xv, y: yv, keep_prob:1.0})
                acc /= counter

                print '#############################################'
                print 'Epoch:', epoch, ' test_acc:', acc
                f.write('Epoch:' + str(epoch) + 'test_acc:' +  str(acc) + '\n')

	    'training per epoch'
	    dt, lt = rshuffle(train_data, train_labels)
#            ind = np.arange(train_data.shape[0])
#            np.random.shuffle(ind)
#            train_data = train_data[ind]
#            train_labels = train_labels[ind]
	    tnorm=0.
	    tloss=0.
	    counter = 0.
            for batch in range(0, dt.shape[0], batch_size):
                xt = dt[batch:batch+batch_size]
                xt = (xt / 255.0) - 0.5
                yt = lt[batch:batch+batch_size]
                # _, norm_, train_loss_ = sess.run([train_step, norm, cross_entropy], feed_dict = {x:xt, y:yt, keep_prob:0.8})
                sess.run(train_step, feed_dict = {x:xt, y:yt, keep_prob:0.8})
                
		if batch ==0: 
		    norm_, train_loss_ = sess.run([norm, cross_entropy], feed_dict = {x:xt, y:yt, keep_prob:1.0})
        	    tnorm += norm_
		    tloss += train_loss_
		    counter += 1.
	    tnorm /= counter
	    tloss /= counter
	    print 'Epoch:', epoch, 'tnorm: ', tnorm, ' tloss: ', tloss
            f.write('Epoch:' + str(epoch) + 'tnorm:' +  str(tnorm) +  'tloss:' + str(tloss)+'\n')

	    'save a snapshot of trained model'
            if epoch % 50 == 0:
                savepath = saver.save(sess, 'model'+str(epoch))
                print 'saving ',savepath
