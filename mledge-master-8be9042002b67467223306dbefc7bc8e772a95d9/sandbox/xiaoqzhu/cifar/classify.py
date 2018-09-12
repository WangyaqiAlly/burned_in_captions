'''
    classify.py:  train and test simply classifier over tiny-images (32x32) 
'''
# CUDA_VISIBLE_DEVICES='0' python classify.py

import argparse
import struct
import tensorflow as tf
import numpy as np
import os
import cv2

classlist = []
classlist.append('airplane')
classlist.append('automobile')
classlist.append('bird')
classlist.append('cat')
classlist.append('deer')
classlist.append('dog')
classlist.append('face')
# classlist.append('frog')
classlist.append('horse')
classlist.append('ship')
# classlist.append('truck')
classlist.append('person')

def load_dataset(basedir, classlist): 

    nimglist = []
    for i, cname in enumerate(classlist): 
 	classdir = os.path.join(basedir, cname)
 	imglist = os.listdir(classdir)
	nimglist.append(len(imglist))

    print 'images per class: ', nimglist
 
    nimg = sum(nimglist)
    d = np.zeros((nimg,32, 32, 3))
    l = np.zeros(nimg)
    idx = 0
    for i, cname in enumerate(classlist): 
 	classdir = os.path.join(basedir, cname)
 	imglist = os.listdir(classdir)
	for imgfile in imglist:
	    bmpfile = os.path.join(classdir, imgfile) 	    
	    img = cv2.imread(bmpfile)
	    print bmpfile, ' | ', img.shape
 	    
	    d[idx,:,:,:]=img
	    l[idx]=i
	    idx = idx + 1

    d = d/255. - 0.5

    return d, l

def rshuffle(d, l): 

    'randomly shuffle dataset'
    ind = np.arange(d.shape[0])
    np.random.shuffle(ind)
    d = d[ind]
    l = l[ind]
    return d, l

def cnet(args,x,reuse=None):
    print 'classifier network, reuse',reuse
    with tf.variable_scope('cnet',reuse=reuse):
        e = tf.layers.conv2d(inputs=x, filters=args.n, kernel_size=5, strides=1,
                             activation=tf.nn.relu, padding='same') ; print e
        e = tf.layers.max_pooling2d(inputs=e, pool_size=[3, 3], strides=2, padding='same'); print e
        e = tf.nn.lrn(e, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75); print e

        e = tf.layers.conv2d(inputs=e, filters=args.n, kernel_size=5, strides=1,
                             activation=tf.nn.relu, padding='same') ; print e
        e = tf.layers.max_pooling2d(inputs=e, pool_size=[3, 3], strides=2, padding='same'); print e
        e = tf.nn.lrn(e, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75); print e

        e = tf.contrib.layers.flatten(e)
        e = tf.layers.dense(inputs=e, units=args.nd*2, activation=None) ; print e
        e = tf.layers.dense(inputs=e, units=args.nd, activation=None) ; print e
        e = tf.layers.dense(inputs=e, units=args.nclass, activation=None) ; print e
    return e

def calc_cmat(args, lt, lp): 

    cmat = np.zeros((args.nclass, args.nclass))
    nimg = len(lt)
    for i in range(nimg):
	itrue = int(lt[i]) 
	ipred = int(lp[i])
	# print 'updating entry for <%d, %d>' % (itrue, ipred)	
	cmat[itrue, ipred] += 1

    nperclass = nimg/args.nclass
    print cmat
    cmat = cmat*1.0/nperclass

    return cmat
# --------- start of main procedures -----------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n', help='number of units per layer', default=64, type=int)
parser.add_argument('--nd', help='number of units in dense layer', default=192, type=int)
parser.add_argument('--m', help='image size', default=32, type=int)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--batch', help='batch size', default=128, type=int)
parser.add_argument('--epochs', help='training epochs', default=100, type=int)
parser.add_argument('--datadir', help='directory of dataset', 
				 default='/home/xiaoqzhu/dataset/cifar-10/cifar-10-imgs')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

args.nclass = len(classlist)

'load data'
traindir = os.path.join(args.datadir, 'train')
testdir = os.path.join(args.datadir, 'test')

d, l = load_dataset(traindir, classlist)
dt, lt = load_dataset(testdir, classlist)

print 'd.shape: ', d.shape, ' l.shape: ', l.shape
print 'dt.shape: ', dt.shape, ' lt.shape: ', lt.shape

'set up NN in tf'
x = tf.placeholder(tf.float32, shape=[None, args.m, args.m, 3])
y = tf.placeholder(tf.int64, shape=[None])

ex = cnet(args,x) # e(x)
eloss = tf.losses.sparse_softmax_cross_entropy(y,ex)
eopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.5,beta2=0.9)
egrads = eopt.compute_gradients(eloss)
etrain = eopt.apply_gradients(egrads)
enorm = tf.global_norm([i[0] for i in egrads])
epred = tf.nn.softmax(ex)

'random shuffle training data once'
d, l = rshuffle(d, l)
# dt, lt = rshuffle(dt, lt)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        rng_state = np.random.get_state()
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(l)

        # train
        el=0.
        en=0.
        t=0.
        for j in range(0,d.shape[0],args.batch):
            _,el_,en_ = sess.run([etrain,eloss,enorm],feed_dict={x:d[j:j+args.batch],y:l[j:j+args.batch]})
            el+=el_
            en+=en_
            t+=1.

        # test
        acc=0.
        tt=0.
	ntest = len(lt)
	lp=np.zeros(ntest)
        for j in range(0,dt.shape[0],args.batch):
            p = sess.run(epred, feed_dict={x:dt[j:j+args.batch]})
            acc += np.mean(np.argmax(p, axis=1) == lt[j:j+args.batch])
            tt += 1.
	    lp[j:j+args.batch]=np.argmax(p,axis=1)

	# print 'lp: ', lp[:10]
        print 'epoch',i,'eloss',el/t,'enorm',en/t,'accuracy',acc/tt
	# cmat = calc_cmat(args, lt, lp)

print 'final testing result: ', len(lp)
'calculate confusion matrix'
cmat = calc_cmat(args, lt, lp)
print cmat
