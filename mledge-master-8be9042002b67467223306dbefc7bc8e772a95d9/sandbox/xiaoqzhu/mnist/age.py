# CUDA_VISIBLE_DEVICES='0' python age.py
import argparse
import struct
import time
import sys
import os
import numpy as np
print 'numpy ' + np.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print 'tensorflow ' + tf.__version__
import cv2
print 'cv2 ' + cv2.__version__
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=10, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default=0.00001, type=float)
parser.add_argument('--clip_ratio', help='gradient clipping', default=100.0, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=1000000, type=int)
parser.add_argument('--model', help='output model', default='model.proto')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

with open('train-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    d = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    d = d/255. - 0.5


print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()
nimg = len(d)

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
        e = tf.layers.dense(inputs=e, units=args.m, activation=tf.tanh) ; print e
        e = tf.identity(e,name='eout') ; print e
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
        g = tf.identity(g,name='gout') ; print g
    return g

x = tf.placeholder('float32', [None,28,28,1],name='x') ; print x
z = tf.placeholder('float32', [None,args.m],name='z') ; print z

ex = enet(args,x) # e(x)
gz = gnet(args,z) # g(z)
egz = enet(args,gz,reuse=True) # e(g(z))

ecost = tf.sqrt(tf.reduce_mean(tf.square((1./args.m)-tf.reduce_mean(np.square(ex),axis=0)))) + tf.sqrt(tf.reduce_mean(tf.square((1./args.m)-tf.reduce_mean(np.square(ex),axis=1))))
gcost = tf.sqrt(tf.reduce_mean(tf.square((1./args.m)-tf.reduce_mean(np.square(egz),axis=0)))) + tf.sqrt(tf.reduce_mean(tf.square((1./args.m)-tf.reduce_mean(np.square(egz),axis=1))))

eopt = tf.train.AdamOptimizer(learning_rate=args.lr)
egrads,evars = zip(*eopt.compute_gradients(ecost-gcost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet')))
egrads,enorm = tf.clip_by_global_norm(egrads, args.clip_ratio)
etrain = eopt.apply_gradients(zip(egrads, evars))

gopt = tf.train.AdamOptimizer(learning_rate=args.lr)
ggrads,gvars = zip(*gopt.compute_gradients(gcost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet')))
ggrads,gnorm = tf.clip_by_global_norm(ggrads, args.clip_ratio)
gtrain = gopt.apply_gradients(zip(ggrads, gvars))

init = tf.global_variables_initializer()

# place-holder figure
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
redline, = ax.plot([], [], 'd', mfc='red', mec='none', ms=2)
blueline, = ax.plot([], [], 'o', mfc='blue', mec='none', ms=2)
greenline, = ax.plot([], [], 's', mfc='green', mec='none', ms=4)
redline.set_data([], [])
blueline.set_data([], [])
greenline.set_data([], [])


with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        np.random.shuffle(d)
        ec=0.
        gc=0.
        en=0.
        gn=0.
        t=0.
        for j in range(0,d.shape[0],args.batch):
            z0 = np.random.randn(args.batch,args.m)
            z0 = z0 / np.sqrt((z0*z0).sum(axis=1))[:,np.newaxis]
            _,ec_,en_ = sess.run([etrain,ecost,enorm],feed_dict={x:d[j:j+args.batch],z:z0})
            z0 = np.random.randn(args.batch,args.m)
            z0 = z0 / np.sqrt((z0*z0).sum(axis=1))[:,np.newaxis]
            _,gc_,gn_ = sess.run([gtrain,gcost,gnorm], feed_dict={z:z0})
            ec+=ec_
            gc+=gc_
            en+=en_
            gn+=gn_
            t+=1.

        print 'epoch',i,'ecost',ec/t,'gcost',gc/t,'enorm',en/t,'gnorm',gn/t

        z0 = np.random.randn(args.batch, args.m)
        z0 = z0 / np.sqrt((z0*z0).sum(axis=1))[:,np.newaxis]
        rid = np.random.randint(nimg, size=args.batch)
        x0 = d[rid]
        zhat = sess.run(ex, feed_dict={x:x0})
        egz0 = sess.run(egz, feed_dict={z:z0})
	print 'egz: ', egz0[0:10, :]
        redline.set_data(z0[:,0], z0[:,1])
        blueline.set_data(zhat[:,0], zhat[:,1])
        greenline.set_data(egz0[:,0], egz0[:,1])
        fig.canvas.draw()

        if args.debug:
            z0 = sess.run(ex,feed_dict={x:d[0:10]})
            for k in range(10):
                print z0[k], np.sum(np.square(z0[k]))

        if i % 100 == 99: 
	    'save generated image to file'
	    imgfile = 'age-%04d.jpg' % i	
	    z0 = np.random.randn(args.batch,args.m)
            z0 = z0 / np.sqrt((z0*z0).sum(axis=1))[:,np.newaxis]
            x0 = sess.run(gz, feed_dict={z:z0})
            x0 = np.clip(x0+0.5,0.,1.)*255.
            img = cv2.resize(np.concatenate((x0[0:10]).astype('uint8'),axis=1),(1000,100))
	    cv2.imwrite(imgfile, img)
  	    # cv2.imshow('img', cv2.resize(np.concatenate((x0[0:10]).astype('uint8'),axis=1),(1000,100)))
            # cv2.waitKey(10)

        # write model, redirect stderr to supress annoying messages
        with open(os.devnull, 'w') as sys.stdout:
            graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout'])
        sys.stdout=sys.__stdout__
        tf.train.write_graph(graph, '.', args.model, as_text=False)
