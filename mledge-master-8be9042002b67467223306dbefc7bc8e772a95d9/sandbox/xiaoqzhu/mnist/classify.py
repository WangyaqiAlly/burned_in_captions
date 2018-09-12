# CUDA_VISIBLE_DEVICES='0' python classify.py
import argparse
import struct
import time
import numpy as np
print 'numpy ' + np.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print 'tensorflow ' + tf.__version__

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=100, type=int)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

with open('train-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    d = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    d = d/255.

with open('train-labels-idx1-ubyte','rb') as f:
    h = struct.unpack('>II',f.read(8))
    l = np.fromstring(f.read(), dtype=np.uint8).astype('int32')

with open('t10k-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    dt = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    dt = dt/255.

with open('t10k-labels-idx1-ubyte','rb') as f:
    h = struct.unpack('>II',f.read(8))
    lt = np.fromstring(f.read(), dtype=np.uint8).astype('int32')

print 'd.shape',d.shape,'l.shape',l.shape,'dt.shape',dt.shape,'lt.shape',lt.shape

def enet(args,x,reuse=None):
    print 'encoder network, reuse',reuse
    with tf.variable_scope('enet',reuse=reuse):
        e = tf.layers.conv2d(inputs=x, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.layers.conv2d(inputs=e, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.image.resize_bilinear(images=e,size=[14,14]) ; print e
        e = tf.layers.conv2d(inputs=e, filters=2*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.layers.conv2d(inputs=e, filters=2*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.image.resize_bilinear(images=e,size=[7,7]) ; print e
        e = tf.layers.conv2d(inputs=e, filters=3*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.layers.conv2d(inputs=e, filters=3*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.contrib.layers.flatten(e)
        e = tf.layers.dense(inputs=e, units=10, activation=None) ; print e
    return e

x = tf.placeholder('float32', [None,28,28,1],name='x') ; print x
y = tf.placeholder('int32', [None],name='y') ; print y

ex = enet(args,x) # e(x)
eloss = tf.losses.sparse_softmax_cross_entropy(y,ex)
eopt = tf.train.AdamOptimizer(learning_rate=args.lr)
egrads = eopt.compute_gradients(eloss)
etrain = eopt.apply_gradients(egrads)
enorm = tf.global_norm([i[0] for i in egrads])
epred = tf.nn.softmax(ex)

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
        for j in range(0,dt.shape[0],args.batch):
            p = sess.run(epred, feed_dict={x:dt[j:j+args.batch]})
            acc += np.mean(np.argmax(p, axis=1) == lt[j:j+args.batch])
            tt += 1.

        print 'epoch',i,'eloss',el/t,'enorm',en/t,'accuracy',acc/tt
