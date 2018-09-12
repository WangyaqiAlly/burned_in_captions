# CUDA_VISIBLE_DEVICES='0' python ali_label.py
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
parser.add_argument('--m', help='latent space dimensionality', default=256, type=int)
parser.add_argument('--n', help='number of units per layer', default=32, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
parser.add_argument('--batch', help='batch size', default=100, type=int)
parser.add_argument('--epochs', help='training epochs', default=20000, type=int)
parser.add_argument('--model', help='output model', default='model.proto.ali_cifar_label_no_embedder')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--bn_training', default=False)
parser.add_argument('--step', help='step to start training', default=0, type=int)
args = parser.parse_args()
print args


tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]

if not os.path.exists(tag):
    os.makedirs(tag)


alpha = 0.1

height = 32
width = 32

import cPickle
parent_path = './CIFAR10_data/'
fileNames = os.listdir(parent_path)
print fileNames
images = None
labels = None
for f in fileNames:
	if 'data_batch' in f:
	    with open(parent_path + '/' + f, 'rb') as fo:
	        tempDict = cPickle.load(fo)
	    tempArr = tempDict['data']
	    tempLabel = tempDict['labels']
	    per_images = np.zeros((len(tempLabel), 32, 32, 3))
	    per_labels = np.zeros(len(tempLabel), dtype=np.int)
	    for i in range(len(tempLabel)):
	        im = np.reshape(tempArr[i], (32, 32, 3), order='F')
	        rows, cols, _ = im.shape
	        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
	        im = cv2.warpAffine(im, M, (cols, rows))
	        per_images[i] = im / 255.0
	        per_labels[i] = tempLabel[i]
	    if images is None:
	        images, labels = per_images, per_labels
	    else:
	        images = np.concatenate([images, per_images], axis=0)
	        labels = np.concatenate([labels, per_labels], axis=0)
print images.shape
d = images[..., [2, 1, 0]]
print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()

nb_classes = 10
targets = np.array([labels]).reshape(-1)
labels = np.eye(nb_classes)[targets]
labels = labels.reshape((-1,1,1,10))
print labels.shape

### Activation function for the discriminator, used together with dropout
def maxOut(x, num_pieces):
	new_shape = x.get_shape().as_list()
	num_channels = new_shape[-1]
	new_shape[-1] = num_channels // num_pieces
	new_shape += [num_pieces]
	new_shape[0] = -1
	x = tf.reshape(x, shape = new_shape)
	y = tf.reduce_max(x, -1, keep_dims = False)
	return y

### Embeds the binary labels
def embedder(args, y):
	print 'embedder'
	with tf.variable_scope('embedder'):
		embeddings = tf.layers.dense(inputs=y,units=64, activation=None); print embeddings
	return embeddings

def enet(args, x, embeddings, reuse=None):
	print 'encoder network, reuse',reuse
	with tf.variable_scope('enet',reuse=reuse):
	    # 32 X 32 X 3 -> 28 X 28 X 32
            e = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, strides=1,data_format='channels_last',activation=None, padding='valid') ; print e
            e = tf.layers.batch_normalization (e, training=args.bn_training)
            e = tf.maximum(alpha*e,e) # LeakyReLU
	    # 28 X 28 X 32 -> 13 X 13 X 64
            e = tf.layers.conv2d(inputs=e, filters=64, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='valid') ; print e
	    e = tf.layers.batch_normalization (e, training=args.bn_training)
	    e = tf.maximum(alpha*e,e) # LeakyReLU
	    # 13 X 13 X 64 -> 10 X 10 X 128
	    e = tf.layers.conv2d(inputs=e, filters=128, kernel_size=4, strides=1,data_format='channels_last',activation=None, padding='valid') ; print e
	    e = tf.layers.batch_normalization (e, training=args.bn_training)
	    e = tf.maximum(alpha*e,e) # LeakyReLU
	    # 10 X 10 X 128 -> 4 X 4 X 256
	    e = tf.layers.conv2d(inputs=e, filters=256, kernel_size=4, strides=2,data_format='channels_last',activation=None, padding='valid') ; print e
	    e = tf.layers.batch_normalization (e, training=args.bn_training)
	    e = tf.maximum(alpha*e,e) # LeakyReLU
	    # 4 X 4 X 256 -> 1 X 1 X 512
	    e = tf.layers.conv2d(inputs=e, filters=512, kernel_size=4, strides=1,data_format='channels_last',activation=None, padding='valid') ; print e
	    e = tf.layers.batch_normalization (e, training=args.bn_training)
	    e = tf.maximum(alpha*e,e) # LeakyReLU
	
            # 1 X 1 X 512 -> 1 X 1 X 552
            pre_z_embed_y = tf.concat([e, embeddings], axis = 3)
            # "Last layer"
            e = tf.layers.conv2d(inputs=pre_z_embed_y, filters=args.m * 2, kernel_size=1, strides=1,data_format='channels_last',activation=None, padding='valid') ; print e
	    e = tf.identity(e,name='eout') ; print e
	    return e

def gaussianConditional(args, e_out):
	mu, log_sigma = e_out[:, :, :, :256], e_out[:, :, :, 256:]
	sigma = tf.exp(log_sigma)
	##############################################
	epsilon = tf.random_normal(tf.shape(mu))
	return mu + sigma * epsilon



def gnet(args,z, embeddings, reuse=None):
	print 'generator network, reuse', reuse
	z_embed_y = tf.concat([z, embeddings], axis = 3)
	z = z_embed_y
	with tf.variable_scope('gnet',reuse=reuse):
		print "Shape of Z:",z.shape
		g = tf.layers.conv2d_transpose(inputs=z, filters=256, kernel_size=4, strides=[1,1], padding='valid', data_format='channels_last', activation=None) ; print g
		g = tf.layers.batch_normalization (g, training=args.bn_training)
		g = tf.maximum(alpha*g,g) #; print g # LeakyReLU
		
		g = tf.layers.conv2d_transpose(inputs=g, filters=128, kernel_size=4, strides=[2,2], padding='valid', data_format='channels_last', activation=None) ; print g
		g = tf.layers.batch_normalization (g, training=args.bn_training)
		g = tf.maximum(alpha*g,g) #; print g # LeakyReLU
		
		g = tf.layers.conv2d_transpose(inputs=g, filters=64, kernel_size=4, strides=[1,1], padding='valid', data_format='channels_last', activation=None) ; print g
		g = tf.layers.batch_normalization (g, training=args.bn_training)
		g = tf.maximum(alpha*g,g) #; print g # LeakyReLU
		
		g = tf.layers.conv2d_transpose(inputs=g, filters=32, kernel_size=4, strides=[2,2], padding='valid', data_format='channels_last', activation=None) ; print g
		g = tf.layers.batch_normalization (g, training=args.bn_training)
		g = tf.maximum(alpha*g,g) #; print g # LeakyReLU
		
		g = tf.layers.conv2d_transpose(inputs=g, filters=32, kernel_size=5, strides=[1,1], padding='valid', data_format='channels_last', activation=None) ; print g
		g = tf.layers.batch_normalization (g, training=args.bn_training)
		g = tf.maximum(alpha*g,g) #; print g # LeakyReLU
		
		g = tf.layers.conv2d_transpose(inputs=g, filters=32, kernel_size=1, strides=[1,1], padding='valid', data_format='channels_last', activation=None) ; print g
		g = tf.layers.batch_normalization (g, training=args.bn_training)
		g = tf.maximum(alpha*g,g) #; print g # LeakyReLU
	        g = tf.layers.conv2d(inputs=g, filters=3, kernel_size=1, strides=1,activation=tf.sigmoid, padding='valid') ; print g
	        g = tf.identity(g,name='gout') ; print g
	return g



def disc_xzy(args, x, z, embeddings, reuse=None):
	print 'D_xz network, reuse', reuse
	with tf.variable_scope('dxzynet',reuse=reuse):
        #### D(x)
		dx = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, strides=1,activation=None, padding='valid') ; print dx
		dx = tf.nn.dropout(dx, 0.8)
		dx = maxOut(dx, 2) 
		dx = tf.layers.conv2d(inputs=dx, filters=64, kernel_size=4, strides=2,activation=None, padding='valid') ; print dx
		dx = tf.nn.dropout(dx, 0.5)
		dx = maxOut(dx, 2) 
		dx = tf.layers.conv2d(inputs=dx, filters=128, kernel_size=4, strides=1,activation=None, padding='valid') ; print dx
		dx = tf.nn.dropout(dx, 0.5)
		dx = maxOut(dx, 2) 	
		dx = tf.layers.conv2d(inputs=dx, filters=256, kernel_size=4, strides=2,activation=None, padding='valid') ; print dx
		dx = tf.nn.dropout(dx, 0.5)
		dx = maxOut(dx, 2) 
		dx = tf.layers.conv2d(inputs=dx, filters=512, kernel_size=4, strides=1,activation=None, padding='valid') ; print dx
		dx = tf.nn.dropout(dx, 0.5)
		dx = maxOut(dx, 2) 

		#### D(z)
		dz = tf.layers.conv2d(inputs=z, filters=512, kernel_size=1, strides=1,activation=None, padding='valid') ; print dz
		dz = tf.nn.dropout(dz, 0.2)
		dz = maxOut(dz, 2)
		dz = tf.layers.conv2d(inputs=dz,filters=512, kernel_size=1, strides=1,activation=None, padding='valid') ; print dz
		dz = tf.nn.dropout(dz, 0.5)
		dz = maxOut(dz, 2)

		#### D(x,z,y)
		xzy = tf.concat(values=[dx,dz,embeddings],axis=3)
		dxzy = tf.layers.conv2d(inputs=xzy, filters=1024, kernel_size=1, strides=1,activation=None, padding='valid') ; print dxzy
		dxzy = tf.nn.dropout(dxzy, 0.5)
		dxzy = maxOut(dxzy, 2)
		dxzy = tf.layers.conv2d(inputs=dxzy, filters=1024, kernel_size=1, strides=1,activation=None, padding='valid') ; print dxzy
		dxzy = tf.nn.dropout(dxzy, 0.5)
		dxzy = maxOut(dxzy, 2)
		dxzy = tf.layers.conv2d(inputs=dxzy, filters=1, kernel_size=1, strides=1,activation=None, padding='valid') ; print dxzy
		dxzy = tf.nn.dropout(dxzy, 0.5)
	
		dxzy = tf.identity(dxzy,name='dxzyout')
	return dxzy



x = tf.placeholder('float32', [None,width,height,3],name='x') ; print x
z = tf.placeholder('float32', [None,1,1,args.m],name='z') ; print z
y = tf.placeholder('float32', [None,1,1,10], name = 'y'); print y

#embeddings = embedder(args,y)
embeddings = y
g_x = enet(args, x, embeddings) # e(x) (z_hat)
g_x = gaussianConditional(args, g_x) # e(x) after normalization

g_z = gnet(args, z, embeddings) # g(z) (x_tilde)

discXZ_hat = disc_xzy(args,x,g_x,embeddings) # D(x,z_hat)
discX_tildeZ = disc_xzy(args,g_z,z,embeddings,reuse=True) # D(x_tilde,z)


dloss = tf.reduce_mean(tf.nn.softplus(tf.negative(discXZ_hat)) + tf.nn.softplus(discX_tildeZ))
gloss = tf.reduce_mean(tf.nn.softplus(discXZ_hat) + tf.nn.softplus(tf.negative(discX_tildeZ)))

dopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1 = 0.5)
v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'dxzynet') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'embedder')
dgrads = dopt.compute_gradients(dloss,var_list=v1)
dtrain = dopt.apply_gradients(dgrads)
dnorm = tf.global_norm([i[0] for i in dgrads])

gopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1 = 0.5)
v2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'embedder')
ggrads = gopt.compute_gradients(gloss,var_list=v2)
gtrain = gopt.apply_gradients(ggrads)
gnorm = tf.global_norm([i[0] for i in ggrads])

eopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1 = 0.5)
v3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'embedder')
egrads = eopt.compute_gradients(gloss,var_list=v3)
etrain = gopt.apply_gradients(egrads)
enorm = tf.global_norm([i[0] for i in egrads])


saver = tf.train.Saver(max_to_keep = 1000)
step = args.step



with tf.Session() as sess:
    if step == 0:
    	sess.run(tf.global_variables_initializer())
    else:
    	model_path = os.path.join(tag, 'model-' + str(step))
    	saver.restore(sess, model_path)
    for i in range(args.epochs):
    	# shuffle both the data and the labels
    	ind = np.arange(d.shape[0])
        np.random.shuffle(ind)
        d = d[ind]
        labels = labels[ind]

        ec=0.
        gc=0.
 	dc=0.
        en=0.
        gn=0.
	dn=0.
        t=0.
        for j in range(0,d.shape[0],args.batch):
		input_x = d[j:j+args.batch]
		input_y = labels[j:j+args.batch]
		input_z = np.random.randn(args.batch,args.m)
		input_z = input_z.reshape(args.batch,1,1,args.m)
	        _,ec_,en_ = sess.run([etrain,gloss,enorm], feed_dict={x:input_x,z:input_z,y:input_y})
	        _,gc_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={x:input_x,z:input_z,y:input_y})
		_,dc_,dn_ = sess.run([dtrain,dloss,dnorm], feed_dict={x:input_x,z:input_z,y:input_y})
		ec=ec_
		gc=gc_
		dc=dc_
		en=en_
		gn=gn_
		dn=dn_
		t=1.
		step += 1
	        if (step % 1000) == 0:
	            savepath = saver.save(sess, os.path.join(tag, 'model'), global_step=step, write_meta_graph=False)
	            print 'saving ',savepath
		print 'epoch',i,'image number',j,'ecost',ec/t,'gcost',gc/t,'dcost',dc/t,'enorm',en/t,'gnorm',gn/t,'dnorm',dn/t
		if j%1000 == 0:
		    xgen = sess.run(g_z, feed_dict={x:input_x, z:np.random.randn(args.batch,1,1,args.m), y:input_y})
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
		graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout', 'dxzynet/dxzyout'])
        sys.stdout=sys.__stdout__
        tf.train.write_graph(graph, '.', args.model, as_text=False)





