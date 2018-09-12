# CUDA_VISIBLE_DEVICES='0' python gan.py
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
import cv2
print 'cv2 ' + cv2.__version__

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=10, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=1000000, type=int)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

with open('train-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    d = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    d = d/255.

print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()

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
        e = tf.layers.dense(inputs=e, units=1, activation=tf.sigmoid) ; print e
    return e

def gnet(args,z,reuse=None):
    print 'generator network, reuse', reuse
    with tf.variable_scope('gnet',reuse=reuse):
        g = tf.layers.dense(inputs=z, units=8*8*args.n, activation=None) ; print g
        g = tf.reshape(g,[-1,8,8,args.n]) ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.image.resize_bilinear(images=g,size=[14,14]) ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.image.resize_bilinear(images=g,size=[28,28]) ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.layers.conv2d(inputs=g, filters=1, kernel_size=3, strides=1,activation=tf.tanh, padding='same') ; print g
    return g

x = tf.placeholder('float32', [None,28,28,1],name='x') ; print x
z = tf.placeholder('float32', [None,args.m],name='z') ; print z

ex = enet(args,x) # e(x)
gz = gnet(args,z) # g(z)
egz = enet(args,gz,reuse=True) # e(g(z))

eloss = -tf.reduce_mean(tf.log(ex) + tf.log(1. - egz))
gloss = -tf.reduce_mean(tf.log(egz))

eopt = tf.train.AdamOptimizer(learning_rate=args.lr)
egrads = eopt.compute_gradients(eloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
etrain = eopt.apply_gradients(egrads)
enorm = tf.global_norm([i[0] for i in egrads])

gopt = tf.train.AdamOptimizer(learning_rate=args.lr)
ggrads = gopt.compute_gradients(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet'))
gtrain = gopt.apply_gradients(ggrads)
gnorm = tf.global_norm([i[0] for i in ggrads])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        np.random.shuffle(d)
        el=0.
        gl=0.
        en=0.
        gn=0.
        t=0.
        for j in range(0,d.shape[0],args.batch):
            _,el_,en_ = sess.run([etrain,eloss,enorm],feed_dict={x:d[j:j+args.batch],z:np.random.randn(args.batch,args.m)})
            zsample = np.random.randn(args.batch,args.m)
            _,gl_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={z:np.random.randn(args.batch,args.m)})
            el+=el_
            gl+=gl_
            en+=en_
            gn+=gn_
            t+=1.
        print 'epoch',i,'eloss',el/t,'gloss',gl/t,'enorm',en/t,'gnorm',gn/t
        x0 = sess.run(gz, feed_dict={z:np.random.randn(args.batch,args.m)})
        x0 = np.clip(x0,0.,1.)*255.
        cv2.imshow('img', cv2.resize(np.concatenate((x0[0:10]).astype('uint8'),axis=1),(1000,100)))
        cv2.waitKey(10)
