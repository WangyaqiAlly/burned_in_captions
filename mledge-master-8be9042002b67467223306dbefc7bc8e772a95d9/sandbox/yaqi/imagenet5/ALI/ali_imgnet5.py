# CUDA_VISIBLE_DEVICES='0' python ali.py
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


import util.tflib as lib
import util.tflib.save_images
import util.tflib.inception_score
import util.tflib.plot
from load_imgnet5 import load_imgnet5_train
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=256, type=int)
parser.add_argument('--n', help='number of units per layer', default=32, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
parser.add_argument('--batch', help='batch size', default=100, type=int)
parser.add_argument('--epochs', help='training epochs', default=10000, type=int)
parser.add_argument('--model', help='output model', default='model.proto.ali.cifar.50')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--bn_training', default=False)
parser.add_argument('--step', help='step to start training', default=0, type=int)
args = parser.parse_args()
print args

alpha = 0.1

height = 32
width = 32
tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]+'_plus'

tag = os.path.join('Record',tag)

if not os.path.exists(tag):
    os.makedirs(tag)

# dataFolder = args.dataFolder
# fileNames = os.listdir(dataFolder)



Figure_dir = os.path.join(tag,'figure')

if not os.path.exists(Figure_dir):
    os.makedirs(Figure_dir)

Sample_dir =os.path.join(tag,'Sample')

if not os.path.exists(Figure_dir):
    os.makedirs(Sample_dir)

# import cPickle
# parent_path = './CIFAR10_data/'
# fileNames = os.listdir(parent_path)
# print fileNames
# images = None
# labels = None
# for f in fileNames:
#   if 'data_batch' in f:
#       with open(parent_path + '/' + f, 'rb') as fo:
#           tempDict = cPickle.load(fo)
#       tempArr = tempDict['data']
#       tempLabel = tempDict['labels']
#       per_images = np.zeros((len(tempLabel), 32, 32, 3))
#       per_labels = np.zeros(len(tempLabel), dtype=np.int)
#       for i in range(len(tempLabel)):
#           im = np.reshape(tempArr[i], (32, 32, 3), order='F')
#           rows, cols, _ = im.shape
#           M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
#           im = cv2.warpAffine(im, M, (cols, rows))
#           per_images[i] = im / 255.0
#           per_labels[i] = tempLabel[i]
#       if images is None:
#           images, labels = per_images, per_labels
#       else:
#           images = np.concatenate([images, per_images], axis=0)
#           labels = np.concatenate([labels, per_labels], axis=0)
# print images.shape
# d = images[..., [2, 1, 0]]

# images = np.load('./data/train/images.npy')
# images = np.reshape(images, (-1, height, width, 3), order = 'F')
# #images = np.transpose(images, (0,2,1,3))
# print images.shape
# d = images/255. #[..., [2, 1, 0]]/255.

d, _ =load_imgnet5_train(datadir='../Data/imgnet5/pfiles',shuffle=False,nt=4)

d = d/255.


print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()
#labels = np.load('./cifar10_50percent/orig_label0.npy')
#assert False

def gaussianConditional(args, e_out):
    mu, log_sigma = e_out[:, :, :, :args.m], e_out[:, :, :, args.m:]
    sigma = tf.exp(log_sigma)
    ##############################################
    epsilon = tf.random_normal(tf.shape(mu))
    return mu + sigma * epsilon


def maxOut(x, num_pieces):
    new_shape = x.get_shape().as_list()
    # print '**********************************'
    # print new_shape
    num_channels = new_shape[-1]
    new_shape[-1] = num_channels // num_pieces
    new_shape += [num_pieces]
    # print '**********************************'
    # print new_shape
    new_shape[0] = -1
    x = tf.reshape(x, shape = new_shape)
    y = tf.reduce_max(x, -1, keep_dims = False)
    return y


### Designed based on ALI paper with some modfiications 
def enet(args,x,reuse=None):
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
        # 1 X 1 X 512 -> 1 X 1 X 512
        e = tf.layers.conv2d(inputs=e, filters=512, kernel_size=1, strides=1,data_format='channels_last',activation=None, padding='valid') ; print e
        e = tf.layers.batch_normalization (e, training=args.bn_training)
        e = tf.maximum(alpha*e,e)

        e = tf.layers.conv2d(inputs=e, filters= 2 * args.m, kernel_size=1, strides=1,data_format='channels_last',activation=None, padding='valid') ; print e
        e = tf.identity(e,name='eout') ; print e
        return e





def gnet(args,z,reuse=None):
    print 'generator network, reuse', reuse
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




def disc_xz(args,x,z,reuse=None):
    print 'D_xz network, reuse', reuse
    with tf.variable_scope('dxznet',reuse=reuse):
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

        #### D(x,z)
        xz = tf.concat(values=[dx,dz],axis=3)
        dxz = tf.layers.conv2d(inputs=xz, filters=1024, kernel_size=1, strides=1,activation=None, padding='valid') ; print dxz
        dxz = tf.nn.dropout(dxz, 0.5)
        dxz = maxOut(dxz, 2)
        dxz = tf.layers.conv2d(inputs=dxz, filters=1024, kernel_size=1, strides=1,activation=None, padding='valid') ; print dxz
        dxz = tf.nn.dropout(dxz, 0.5)
        dxz = maxOut(dxz, 2)
        dxz = tf.layers.conv2d(inputs=dxz, filters=1, kernel_size=1, strides=1,activation=None, padding='valid') ; print dxz
        dxz = tf.nn.dropout(dxz, 0.5)
        dxz = tf.identity(dxz,name='dxzout')
        return dxz



x = tf.placeholder('float32', [None,width,height,3],name='x') ; print x
z = tf.placeholder('float32', [None,1,1,args.m],name='z') ; print z

g_x = enet(args,x) # e(x)
g_x = gaussianConditional(args, g_x)
g_z = gnet(args,z) # g(z)
discXZ_hat = disc_xz(args,x,g_x) # D(x,z_hat)
discX_tildeZ = disc_xz(args,g_z,z,reuse=True) # D(x_tilde,z)



dloss = tf.reduce_mean(tf.nn.softplus(tf.negative(discXZ_hat)) + tf.nn.softplus(discX_tildeZ))
gloss = tf.reduce_mean(tf.nn.softplus(discXZ_hat) + tf.nn.softplus(tf.negative(discX_tildeZ)))

dopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
dgrads = dopt.compute_gradients(dloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'dxznet'))
dtrain = dopt.apply_gradients(dgrads)
dnorm = tf.global_norm([i[0] for i in dgrads])

gopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
ggrads = gopt.compute_gradients(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet'))
gtrain = gopt.apply_gradients(ggrads)
gnorm = tf.global_norm([i[0] for i in ggrads])

eopt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1= 0.5)
egrads = eopt.compute_gradients(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
etrain = gopt.apply_gradients(egrads)
enorm = tf.global_norm([i[0] for i in egrads])


saver = tf.train.Saver(max_to_keep = 10000)
step = args.step

# Function for calculating inception score
#fake_labels_100 = tf.cast(tf.random_uniform([100]) * 10, tf.int32)
#samples_100 = Generator(100, fake_labels_100)


fixed_z = np.random.randn(args.batch,1,1,args.m)


with tf.Session() as sess:
    if step == 0:
        sess.run(tf.global_variables_initializer())
    else:
        model_path = os.path.join(tag, 'model-' + str(step))
        saver.restore(sess, model_path)

    xgen_inception = []
    xgen_cnt = 0

    for i in range(args.epochs):
        np.random.shuffle(d)
        ec=0.
        gc=0.
        dc=0.
        en=0.
        gn=0.
        dn=0.
        t=0.
        for j in range(0,d.shape[0] - args.batch, args.batch):
            input_x = d[j:j+args.batch]
            input_z = np.random.randn(args.batch,args.m)
            input_z = input_z.reshape(args.batch,1,1,args.m)
            _,ec_,en_ = sess.run([etrain,gloss,enorm], feed_dict={x:input_x,z:input_z})
            _,gc_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={x:input_x,z:input_z})
            _,dc_,dn_ = sess.run([dtrain,dloss,dnorm], feed_dict={x:input_x,z:input_z})
            ec=ec_
            gc=gc_
            dc=dc_
            en=en_
            gn=gn_
            dn=dn_
            t=1.

            if ((i % 10) == 0) and (j == 0):
                savepath = saver.save(sess, os.path.join(tag, 'model-'+str(i)), global_step=step, write_meta_graph=False)
                print 'saving ',savepath

            if step%10 == 0:
                print 'epoch', i, 'iter', step, 'ecost', ec / t, 'gcost', gc / t, 'dcost', dc / t, 'enorm', en / t, 'gnorm', gn / t, 'dnorm', dn / t
                lib.plot.plot(Figure_dir + '//' + 'e_cost', ec / t)
                lib.plot.plot(Figure_dir + '//' + 'gcost', gc/t)
                lib.plot.plot(Figure_dir + '//' + 'dcost', dc/t)
                lib.plot.plot(Figure_dir + '//' + 'enorm', en/t)
                lib.plot.plot(Figure_dir + '//' + 'gnorm', gn/t)
                lib.plot.plot(Figure_dir + '//' + 'dnorm', dn/t)
                xgen = sess.run(g_z, feed_dict={z:fixed_z})
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
                if step % 100 == 0:

                    pngfname = "%s-%d-%d.png" % (tag, i, j)
                    pathname = os.path.join(Sample_dir, pngfname)
                    cv2.imwrite(pathname, theImage)
                lib.plot.flush()
            if step % 5000 ==0:
                all_samples = []
                for k in xrange(500):
                    all_samples.append(sess.run(g_z, feed_dict={z:np.random.randn(args.batch,1,1,args.m)}))
                all_samples = np.concatenate(all_samples, axis=0)
                all_samples = (np.clip(all_samples,0.,1.)*255.).astype('int32')
                inception_score = lib.inception_score.get_inception_score(list(all_samples))
                lib.plot.plot(Figure_dir + '//' + 'inception_50k', inception_score[0])
                lib.plot.plot(Figure_dir + '//' + 'inception_50k_std', inception_score[1])
                print "inception score at iter {}: mean:{}   var:{} ".format(step, inception_score[0],inception_score[1])
                lib.plot.flush()
            if step < 500 :
                lib.plot.flush()
            lib.plot.tick()
            step += 1


        # write model, redirect stderr to supress annoying messages
        with open(os.devnull, 'w') as sys.stdout:
            graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout', 'dxznet/dxzout'])
            sys.stdout=sys.__stdout__
            tf.train.write_graph(graph, '.', args.model, as_text=False)

