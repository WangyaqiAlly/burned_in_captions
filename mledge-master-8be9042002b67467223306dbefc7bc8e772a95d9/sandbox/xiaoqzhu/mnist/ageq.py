# CUDA_VISIBLE_DEVICES='0' python aqeq.py

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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nz', help='latent space dimensionality', default=100, type=int)
parser.add_argument('--nc', help='number of channels of input', default=1, type=int)
parser.add_argument('--nef', help='factor of number of channels in encoder', default=64, type=int)
parser.add_argument('--ngf', help='factor of number of channels in generator', default=64, type=int)
parser.add_argument('--ne_updates', help='number of updates for encoder', default=1, type=int)
parser.add_argument('--ng_updates', help='number of updates for generator', default=1, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0002, type=float)
parser.add_argument('--batch', help='batch size', default=500, type=int)
parser.add_argument('--epochs', help='training epochs', default=200, type=int)
parser.add_argument('--model', help='output model', default='model.proto.age')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--step', help='step to start training', default=0, type=int)
parser.add_argument('--dataFolder', help='data folder', default='../celebA/img_align_celeba/')
args = parser.parse_args()
print args


tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]

if not os.path.exists(tag):
    os.makedirs(tag)

#dataFolder = args.dataFolder
#fileNames = os.listdir(dataFolder)

alpha = 0.2

height = 32
width = 32



# ######################## Load the Data
with open('./train-labels-idx1-ubyte', 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
with open('./train-images-idx3-ubyte', 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols,1)
d = np.zeros((img.shape[0], height, width, 1))
d[:,2:30, 2:30,:] = img[:,:,:,:]
d = d.astype('float32')
d = d/255.
# dan added scaling
d = (d-0.5)/0.5
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# d = tf.reshape(mnist.train.images, [-1, 28, 28, 1])

# numData = 200000
# d = []
# counter = 0
# for f in fileNames:
#     if '.jpg' not in f:
#         continue
#     counter += 1
#     if counter >numData:
#         break
#     img = cv2.imread(dataFolder+f,cv2.IMREAD_COLOR)
#     img = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)
#     d.append(img)


# d = np.array(d[0:numFaces]).reshape(numFaces,width,height,3).astype('float32')
# d = d/255.

# print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()



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

# KL term: m^2+s^2-log(s^2)
# dtan2: use + square...
def kl_loss(args, samples, direction = 'pq', minimize=True):
    s = tf.contrib.layers.flatten(samples)
    mean, var = tf.nn.moments(s, axes = [0])
    t1 = (tf.pow(mean,2) + var)
    t2 = -tf.log(var)
    kl = tf.reduce_mean(t1 + t2)
    return kl

def kl_loss2(args, samples, direction = 'pq', minimize = True):
	s = tf.contrib.layers.flatten(samples)
	mean, var = tf.nn.moments(s, axes = [0])
	t1 = (1 + tf.pow(mean,2)) / (2 * tf.pow(var,2))
	t2 = tf.log(var)
	kl = tf.reduce_mean(t1 + t2 - 0.5)
	return kl
	# if direction == 'pq':
	# 	t1 = (1 + tf.pow(mean,2)) / (2 * tf.pow(var,2))
	# 	t2 = tf.log(var)
	# 	kl = tf.reduce_mean(t1 + t2 - 0.5)
	# else:
	# 	t1 = (tf.pow(mean,2) + tf.pow(var,2)) / 2
	# 	t2 = -tf.log(var)
	# 	kl = tf.reduce_mean(t1 + t2 - 0.5)
	# if not minimize:
	# 	kl *= -1.0
	# return kl


# compute distance between corresponding points in x and y using dist
# dist = 'L2' / 'L1' / 'cos'
def match(x, y, dist = 'L1'):
	if dist == 'L2':
		return tf.reduce_mean(tf.pow(x-y, 2))
	elif dist == 'L1':
		return tf.reduce_mean(tf.abs(x-y))
	elif dist == 'cos':
		x_norm = normalize(x)
		y_norm = normalize(y)
		return 2.0 - tf.reduce_mean(x_norm * y_norm)
	else:
		return 0.

# projects points to a sphere
def normalize(x, axis = 3):
	x_norm = tf.norm(x, axis = axis)
	#x_norm = tf.tile(x_norm, tf.shape(x_norm))
	return x / x_norm 




######################## Build the Graph
x = tf.placeholder('float32', [None,width,height,1],name='x') ; print x
z = tf.placeholder('float32', [None,1,1,args.nz],name='z') ; print z

e_x = enet(args, x) # e(x)
g_e_x = gnet(args, e_x) # g(e(x)) 

g_z = gnet(args, z, reuse = True)
e_g_z = enet(args, g_z, reuse = True) # e(g(z))

kl_real = kl_loss(args, e_x, minimize = True) ; print kl_real
# dtan2: changed to use negative
kl_fake_e = -kl_loss(args, e_g_z, minimize = False) ; print kl_fake_e
kl_fake_g = kl_loss(args, e_g_z, minimize = True) ; print kl_fake_g

eloss = tf.add(kl_real, kl_fake_e); print eloss # + match(g_e_x, x, 'L1') * 0 + match(e_g_z, z, 'cos') * 0
gloss = kl_fake_g; print gloss#+ match(e_g_z, z, 'cos') * 1

eopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
egrads = eopt.compute_gradients(eloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
etrain = eopt.apply_gradients(egrads)
enorm = tf.global_norm([i[0] for i in egrads])

gopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
ggrads = gopt.compute_gradients(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet'))
gtrain = gopt.apply_gradients(ggrads)
gnorm = tf.global_norm([i[0] for i in ggrads])

saver = tf.train.Saver()
step = args.step


######################## Train the Model
with tf.Session() as sess:
    if step == 0:
    	sess.run(tf.global_variables_initializer())
    else:
    	model_path = os.path.join(tag, 'model-' + str(step))
    	saver.restore(sess, model_path)
    for i in range(args.epochs):

    	ind = np.arange(d.shape[0])
        np.random.shuffle(ind)
        d = d[ind]
	#d = tf.random_shuffle(d)

        for j in range(0,d.shape[0],args.batch):
        	#print j
        	input_x = d[j:j+args.batch]

        	# train the encoder
        	for _ in range(args.ne_updates):
        		input_z = np.random.randn(args.batch,args.nz)
        	        input_z = input_z.reshape(args.batch,1,1,args.nz)
        		_, el, en = sess.run([etrain,eloss,enorm], feed_dict={x:input_x,z:input_z})
                        # print "####################x.shape: ", input_x.shape
        	# train the generator
        	for _ in range(args.ng_updates):
        		input_z = np.random.randn(args.batch,args.nz)
        	        input_z = input_z.reshape(args.batch,1,1,args.nz)
                        # print '##################z.shape: ', input_z.shape
        		_, gl, gn = sess.run([gtrain,gloss,gnorm], feed_dict={x:input_x,z:input_z})
        	

        	t=1.
                step += 1
	        if (step % 50) == 0:
	            savepath = saver.save(sess, os.path.join(tag, 'model'), global_step=step, write_meta_graph=False)
	            print 'saving ',savepath
		    print 'epoch',i,'image number',j,'ecost',el/t,'gcost',gl/t,'enorm',en/t,'gnorm',gn/t
		if j%1000 == 0:
		    xgen = sess.run(g_z, feed_dict={x:input_x, z:np.random.randn(args.batch,1,1,args.nz)})
	            xgen = np.clip(xgen,0.,1.)*255.
                    realImg = np.clip(d[j:j+10],0., 1.)*255
	            myImg1 = np.concatenate((xgen[0:10]).astype('uint8'),axis=1)
	            myImg2 = np.concatenate((realImg).astype('uint8'),axis=1)
	            myImg = np.concatenate((myImg1,myImg2),axis=0)
	            theImage = cv2.resize(myImg,(1000,200))
	            cv2.imshow('img',theImage)
	            k = cv2.waitKey(100)
	            if k==1114083: # ctrl-c to exit
	                break
                    pngfname = "%s-%d-%d.png" % (tag, i, j)
		    pathname = os.path.join(tag, pngfname)
		    cv2.imwrite(pathname, theImage)
		

        # write model, redirect stderr to supress annoying messages
        with open(os.devnull, 'w') as sys.stdout:
		    graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout'])
        sys.stdout=sys.__stdout__
        tf.train.write_graph(graph, '.', args.model, as_text=False)





