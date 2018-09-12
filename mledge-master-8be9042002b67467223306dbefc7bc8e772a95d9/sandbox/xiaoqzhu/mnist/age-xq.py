# CUDA_VISIBLE_DEVICES='0' python age-xq.py --epochs 50
'''
    age2.py: modified cost function 
'''

import argparse
import struct
import time
import sys
import os
import numpy as np
print 'numpy ' + np.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.4f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print 'tensorflow ' + tf.__version__
import cv2
print 'cv2 ' + cv2.__version__

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=10, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--nz', help='latent space dimensionality', default=100, type=int)
parser.add_argument('--nc', help='number of channels of input', default=1, type=int)
parser.add_argument('--nef', help='factor of number of channels in encoder', default=64, type=int)
parser.add_argument('--ngf', help='factor of number of channels in generator', default=64, type=int)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=50, type=int)
parser.add_argument('--model', help='output model', default='age2.model.proto')
parser.add_argument('--vlambda', help='regularizing parameter for z: lambda', default=100.0, type=float)
parser.add_argument('--vmu', help='regularizing parameter for x: mu', default=1.0, type=float)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

'additional global parameters'
alpha = 0.2
height = 32
width = 32

'load data and cast into 32x32 images (centered)'
with open('train-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    img = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')

d = np.zeros((img.shape[0], height, width, 1))
d[:,2:30, 2:30,:] = img[:,:,:,:]
d = d.astype('float32')
d = d/255. - 0.5

print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()
nimg = len(d)
print '# of images in dataset: ', nimg

### Designed based on AGE-GAN paper
def enet(args, x, reuse=None):
    print 'encoder network, reuse',reuse
    with tf.variable_scope('enet',reuse=reuse):
     	# 32 X 32 X 1 -> 16 X 16 X nef
        e = tf.layers.conv2d(inputs=x, filters=args.nef, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='same') ; print e
	e = tf.maximum(alpha*e,e) # LeakyReLU
        
	# 16 X 16 X nef -> 8 X 8 X nef * 2
        e = tf.layers.conv2d(inputs=e, filters=args.nef*2, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='same') ; print e        
	e = tf.layers.batch_normalization (e, training=True)
        e = tf.maximum(alpha*e,e) # LeakyReLU
                
	# 8 X 8 X nef*2 -> 4 X 4 X nef * 4
        e = tf.layers.conv2d(inputs=e, filters=args.nef*4, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='same') ; print e
        e = tf.layers.batch_normalization (e, training=True)
        e = tf.maximum(alpha*e,e) # LeakyReLU
        
	# 4 X 4 X nef * 4 -> 2 X 2 X nz
        e = tf.layers.conv2d(inputs=e, filters=args.nz, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='same') ; print e
                
	# 2 X 2 X nz -> 1 X 1 X nz
        e = tf.nn.pool(e, window_shape = [2,2], pooling_type = 'AVG', padding = 'VALID')
        e = tf.identity(e,name='eout') ; print e
    
	# project to unit sphere
	e = tf.nn.l2_normalize(e, dim=3, name='eout0'); print e
            
    return e

def gnet(args, z, reuse=None):
    print 'generator network, reuse', reuse
    with tf.variable_scope('gnet',reuse=reuse):
	print "Shape of Z:",z.shape
        # 1 X 1 X nz -> 4 X 4 X ngf*8 
        g = tf.layers.conv2d_transpose(inputs=z, filters=args.ngf*8, kernel_size=4, strides=[1,1], padding='valid', data_format='channels_last', activation=None) ; print g
        g = tf.layers.batch_normalization (g, training=True)
        g = tf.nn.elu(g) #; print g 
        
	# 4 X 4 X ngf*8 -> 8 X 8 X ngf*4
        g = tf.layers.conv2d_transpose(inputs=g, filters=args.ngf*4, kernel_size=4, strides=[2,2], padding='same', data_format='channels_last', activation=None) ; print g
        g = tf.layers.batch_normalization (g, training=True)
        g = tf.nn.elu(g)
                
	# 8 X 8 X ngf*4 -> 16 X 16 X ngf*2
        g = tf.layers.conv2d_transpose(inputs=g, filters=args.ngf*2, kernel_size=4, strides=[2,2], padding='same', data_format='channels_last', activation=None) ; print g
        g = tf.layers.batch_normalization (g, training=True)
        g = tf.nn.elu(g)
                
	# 16 X 16 X ngf*2 -> 32 X 32 X ngf*2
        g = tf.layers.conv2d_transpose(inputs=g, filters=args.ngf*2, kernel_size=4, strides=[2,2], padding='same', data_format='channels_last', activation=None) ; print g
        g = tf.layers.batch_normalization (g, training=True)
        g = tf.nn.elu(g)
                
	# 32 X 32 X ngf*2 -> 32 X 32 X nc
        g = tf.layers.conv2d(inputs=g, filters=args.nc, kernel_size=4, strides=1,data_format='channels_last',activation=None, padding='same') ; print g
        g = tf.nn.tanh(g)
        g = tf.identity(g,name='gout') ; print g
        
    return g

def enet2(args,x,reuse=None):
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
        e = tf.layers.dense(inputs=e, units=args.m, activation=None) ; print e
    	# e = tf.nn.l2_normalize(e, 1)
	e = tf.identity(e,name='eout') ; print e
    return e

def gnet2(args,z,reuse=None):
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

def get_z0(args, nsample):
    z0 = np.random.randn(nsample,args.nz)
    z0 = z0 / np.sqrt((z0*z0).sum(axis=1))[:,np.newaxis]
    z0 = z0.reshape(nsample,1,1,args.nz)
    # print 'Shape of z0: ', z0.shape
    return z0


x = tf.placeholder('float32', [None,width,height,1],name='x') ; print x
z = tf.placeholder('float32', [None,1, 1, args.nz],name='z') ; print z

ex = enet(args,x) # e(x)
gz = gnet(args,z) # g(z)
egz = enet(args,gz,reuse=True) # e(g(z))
gex = gnet(args,ex,reuse=True) # g(e(x))

xcost = tf.reduce_mean(tf.norm(x-gex)); print 'xcost: ', xcost
zcost = tf.reduce_mean(tf.norm(z-egz)); print 'zcost: ', zcost

# compoment-wise mean and variance for z
mean, var = tf.nn.moments(tf.contrib.layers.flatten(z), axes = [0])
z_m = tf.reduce_mean(z, axis=0)
z_s2 = tf.reduce_mean(tf.square(z-z_m), axis=0)
z_m2 = tf.square(z_m)
z_kl = tf.reduce_mean(z_s2+z_m2-tf.log(z_s2))

# component-wise mean and variance for ex
ex_m = tf.reduce_mean(ex, axis=0); 
ex_s2 = tf.reduce_mean(tf.square(ex-ex_m), axis=0);
ex_m2 = tf.square(ex_m);
ex_kl = tf.reduce_mean(ex_s2+ex_m2-tf.log(ex_s2))

# component-wise mean, variance for egz
egz_m = tf.reduce_mean(egz, axis=0)
egz_s2 = tf.reduce_mean(tf.square(egz-egz_m), axis=0)
egz_m2 = tf.square(egz_m)
egz_kl = tf.reduce_mean(egz_s2+egz_m2-tf.log(egz_s2))

# ecost = ex_kl - egz_kl + args.vmu*xcost
# gcost = egz_kl + args.vlambda*zcost
gcost = egz_kl
ecost = ex_kl

# print 'gcost: ', gcost
# print 'ecost: ', ecost

eopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
egrads,evars = zip(*eopt.compute_gradients(ecost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet')))
etrain = eopt.apply_gradients(zip(egrads, evars))
enorm = tf.global_norm(egrads)

egrads2,evars2 = zip(*eopt.compute_gradients(-gcost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet')))
etrain2 = eopt.apply_gradients(zip(egrads2, evars2))
enorm2 = tf.global_norm(egrads2)


gopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
ggrads,gvars = zip(*gopt.compute_gradients(gcost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet')))
gtrain = gopt.apply_gradients(zip(ggrads, gvars))
gnorm = tf.global_norm(ggrads)

init = tf.global_variables_initializer()

# place-holder figure
xidx = np.arange(-args.nz, args.nz)
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-args.nz, args.nz), ylim=(-.1, .1))
blueline, = ax.plot([], [], '-d', color='blue', mfc='blue', mec='none', ms=6, label='prior')
redline, = ax.plot([], [], '-o', color='red', mfc='red', mec='none', ms=6, label='fake')
greenline, = ax.plot([], [], '-s', color='green', mfc='green', mec='none', ms=6, label='real')
blueline.set_data([], [])
redline.set_data([], [])
greenline.set_data([], [])
plt.legend()

with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        np.random.shuffle(d)
        ec=0.
        gc=0.
        en=0.
        gn=0.
	xc=0.
	zc=0.
	zd=0.
        t=0.
        for j in range(0,d.shape[0],args.batch):
	    'obtain batch of real & fake data'	   
	    z0 = get_z0(args, args.batch)
	    x0 = d[j:j+args.batch]
	
	    # for k in range(0, 5): 
	    'train generator on fake data'            
	    _,gc_,gn_ = sess.run([gtrain,gcost,gnorm], feed_dict={z:z0})

	    'train encoder on real data'
            _,ec_,en_ = sess.run([etrain,ecost,enorm],feed_dict={x:x0})
		
	    'train encoder on fake data' 
            # z0 = get_z0(args)
            _,ec_,en_ = sess.run([etrain2,-gcost,enorm2],feed_dict={z:z0})
	    
	    'eval component-wise mean and variance'
	    z0_m, z0_s2, z0_kl = sess.run([z_m, z_s2, z_kl], feed_dict={z:z0})
	    z1_m, z1_s2, z1_kl = sess.run([egz_m, egz_s2, egz_kl], feed_dict={z:z0})
	    z2_m, z2_s2, z2_kl = sess.run([ex_m, ex_s2, ex_kl], feed_dict={x:x0})
	
	    z0_val = np.concatenate((z0_m, z0_s2)) 
	    z1_val = np.concatenate((z1_m, z1_s2)) 
	    z2_val = np.concatenate((z2_m, z2_s2)) 
	    
	    blueline.set_data(xidx, z0_val)
            redline.set_data(xidx, z1_val)
	    greenline.set_data(xidx, z2_val)
	    fig.canvas.draw()
	    
	    print 'i = %6d, j=%6d,    KL: %6.2f  | %6.2f  | %6.2f ' % (i, j, z0_kl, z1_kl, z2_kl)
	    
	    ec+=ec_
            gc+=gc_
            en+=en_
            gn+=gn_
            t+=1.

	    # xc_ = sess.run(xcost, feed_dict={x:x0})
	    # zc_ = sess.run(zcost, feed_dict={z:z0})
	    # xc += xc_
	    # zc += zc_
	    # print 'epoch %d, batch %d: ||z|| =%.2f, ||||'	
        
	print 'epoch',i,'ecost',ec/t,'gcost',gc/t, 'enorm',en/t,'gnorm',gn/t
        # print 'epoch', i, 'ecost ', ec/t, 'gcost', gc/t, 'xcost', xc/t, 'zcost', zc/t
        if args.debug:
            z0 = sess.run(ex,feed_dict={x:d[0:10]})
            for k in range(10):
                print z0[k], np.sum(np.square(z0[k]))

	if i >0 : 
	    'save generated image to file'
	    imgfile = 'age2-%04d.jpg' % i
	    nsample = 10	
	
	    'x0: random generated'
	    z0 = get_z0(args, nsample)
            x0 = sess.run(gz, feed_dict={z:z0})
            x0 = np.clip(x0+0.5,0.,1.)*255.

	    'x1: randomly selected original samples'
            rid = np.random.randint(nimg, size=nsample)
	    x1 = d[rid]
	    x1 = np.clip(x1+0.5, 0., 1.)*255.
	    
	    'y0, y1: reconstructed'
	    z1 = sess.run(ex, feed_dict={x:x1})
	    y0 = sess.run(gz, feed_dict={z:z1})
	    y0 = np.clip(y0+0.5, 0., 1.)*255.

	    y1 = sess.run(gex, feed_dict={x:x1})
	    y1 = np.clip(y1+0.5, 0., 1.)*255.

	    img1 = np.concatenate(x0.astype('uint8'), axis=1)
	    img2 = np.concatenate(x1.astype('uint8'), axis=1)
	    img3 = np.concatenate(y0.astype('uint8'), axis=1)
	    img4 = np.concatenate(y1.astype('uint8'), axis=1)
	    img = cv2.resize(np.concatenate((img1, img2, img3, img4), axis=0), (1000, 400))
	    cv2.imwrite(imgfile, img)

        # write model, redirect stderr to supress annoying messages
        with open(os.devnull, 'w') as sys.stdout:
            graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout'])
        sys.stdout=sys.__stdout__
        tf.train.write_graph(graph, '.', args.model, as_text=False)
   
