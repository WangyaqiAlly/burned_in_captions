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
parser.add_argument('--nz', help='latent space dimensionality', default=10, type=int)
parser.add_argument('--nh', help='embedding space dimensionality', default=10, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
parser.add_argument('--lk', help='learning rate for k', default=0.001, type=float)
parser.add_argument('--gamma', help='target diversity ratio', default=0.5, type=float)
parser.add_argument('--batch', help='batch size', default=16, type=int)
parser.add_argument('--epochs', help='training epochs', default=10, type=int)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

with open('train-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    d = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    d = d/255.

print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()


def get_z0(args, nsample):
    z0 = np.random.uniform(-1., 1., (nsample,args.nz))
    # z0 = z0.reshape(nsample,1,1,args.nz)
    # print 'Shape of z0: ', z0.shape
    return z0

def enet(args,x,reuse=None):
    print 'encoder/discriminator network, reuse',reuse
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
        e = tf.layers.dense(inputs=e, units=args.nh, activation=None) ; print e
    return e

def dnet(args,h,reuse=None):
    print 'decoder network, reuse', reuse
    with tf.variable_scope('dnet',reuse=reuse):
        g = tf.layers.dense(inputs=h, units=8*8*args.n, activation=None) ; print g
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
h = tf.placeholder('float32', [None,args.nh],name='h') ; print h
z = tf.placeholder('float32', [None,args.nz],name='z') ; print z
# k = tf.placeholder('float32', [None,1],name='k') ; print k
k = tf.Variable(0.)

ex = enet(args,x) # e(x)
xhat = dnet(args, ex) # d(h) = d(e(x))

gz = gnet(args,z) # g(z)
htilde = enet(args,gz,reuse=True) # e(g(z))
xtilde = dnet(args, htilde, reuse=True) # d(e(g(z))

 
xloss = tf.reduce_mean(tf.abs(xhat-x))
zloss = tf.reduce_mean(tf.abs(xtilde-gz))

ecost = xloss-k*zloss
gcost = zloss
'optimizer for encoder'
eopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
egrads,evars = zip(*eopt.compute_gradients(ecost,
		    	var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet')))
etrain = eopt.apply_gradients(zip(egrads, evars))
enorm = tf.global_norm(egrads)

'optimizer for decoder'
dopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
dgrads,dvars = zip(*dopt.compute_gradients(ecost,
		   	var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'dnet')))
dtrain = dopt.apply_gradients(zip(dgrads, dvars))
dnorm = tf.global_norm(egrads)

'optimizer for generator'
gopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
ggrads,gvars = zip(*gopt.compute_gradients(gcost,
			var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet')))
gtrain = gopt.apply_gradients(zip(ggrads, gvars))
gnorm = tf.global_norm(ggrads)

'update of k'
kopt = k + args.lk*(args.gamma*xloss-zloss)

init = tf.global_variables_initializer()
k0 = 0.
mlist = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        np.random.shuffle(d)
        for j in range(0,d.shape[0],args.batch):
                      
	    'obtain batch of real & fake data'
            z0 = get_z0(args, args.batch)
            x0 = d[j:j+args.batch] 
	    
	    'train auto-encoder'
	    _,xl_,en_ = sess.run([etrain,xloss,enorm],feed_dict={x:x0,z:z0,k:k0})
	    _,xl_,dn_ = sess.run([dtrain,xloss,dnorm],feed_dict={x:x0,z:z0,k:k0})

	    'train generator'
	    z0 = get_z0(args, args.batch)
            _,zl_,gn_ = sess.run([gtrain,zloss,gnorm], feed_dict={z:z0})

	    'calculate global convergence measure'
	    mval = xl_ + np.abs(args.gamma*xl_ - zl_)
 	    mlist.append(mval)
	    'update scaling parameter k'
	    k_ = sess.run(k, feed_dict={k:k0})
	    k0 = k_

            print 'epoch: %d - %d' %(i,j), ' mval: ',mval,' xloss: ',xl_, 'zloss: ', zl_, ' k=', k_
   
'''     
	if i % 100 == 99: 
	    imgfile = 'gan-%04d.jpg' % i
	    x0 = sess.run(gz, feed_dict={z:np.random.randn(args.batch,args.m)})
            x0 = np.clip(x0,0.,1.)*255.
            img = cv2.resize(np.concatenate((x0[0:10]).astype('uint8'), axis=1), (1000, 100))
	    cv2.imwrite(imgfile, img)
	    # cv2.imshow('img', cv2.resize(np.concatenate((x0[0:10]).astype('uint8'),axis=1),(1000,100)))
            # cv2.waitKey(10)
'''
