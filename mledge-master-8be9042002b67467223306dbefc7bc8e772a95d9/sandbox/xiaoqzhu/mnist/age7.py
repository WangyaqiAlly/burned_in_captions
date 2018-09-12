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

tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]

default_model = tag + '.proto'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=10, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default=0.00001, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=5000, type=int)
parser.add_argument('--model', help='output model', default=default_model)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

if not os.path.exists(tag):
    os.makedirs(tag)

with open('train-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    d = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    d = d/255.

print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()
nimg = len(d)
print '# of images in dataset: ', nimg

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
gex = gnet(args, ex,reuse=True)

#ecost = tf.reduce_mean(tf.square(ex)) ; print ecost
me = tf.reduce_mean(ex, axis=0)
se = tf.reduce_mean(tf.square(ex - me), axis=0)
ecost_real = 0.5*tf.reduce_sum(se + tf.square(me) - 2*tf.log(se))
#gcost = tf.reduce_mean(tf.square(egz)) ; print gcost
mg = tf.reduce_mean(egz, axis=0)
sg = tf.reduce_mean(tf.square(egz - mg), axis=0)
gcost = 0.5*tf.reduce_sum(sg + tf.square(mg) - 2*tf.log(sg)); print gcost

zcost = tf.reduce_mean(tf.square(z - egz))
xcost = tf.reduce_mean(tf.square(x - gex))

eopt = tf.train.AdamOptimizer(learning_rate=args.lr)
egrads_real = eopt.compute_gradients(ecost_real + 10*xcost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
etrain_real = eopt.apply_gradients(egrads_real)
enorm_real = tf.global_norm([i[0] for i in egrads_real])
egrads_fake = eopt.compute_gradients(-gcost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
etrain_fake = eopt.apply_gradients(egrads_fake)
enorm_fake = tf.global_norm([i[0] for i in egrads_fake])



gopt = tf.train.AdamOptimizer(learning_rate=args.lr)
ggrads = gopt.compute_gradients(gcost + 1000*zcost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet'))
gtrain = gopt.apply_gradients(ggrads)
gnorm = tf.global_norm([i[0] for i in ggrads])

init = tf.global_variables_initializer()

def sphere(a,b):
    z0 = np.random.randn(a,b)
    s = np.sqrt(np.sum(z0*z0,axis=1))
    for i in range(a):
        z0[i,:] /= s[i]
    return z0

with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        np.random.shuffle(d)
        ec=0.
        gc=0.
        en=0.
        gn=0.
        ec2=0.
        en2=0.
        t=0.
        for j in range(0,d.shape[0],args.batch):
            z0 = sphere(args.batch,args.m)
            _,ec_,en_ = sess.run([etrain_real,ecost_real,enorm_real],feed_dict={x:d[j:j+args.batch]})
            # train generator twice
            sess.run([gtrain,gcost,gnorm], feed_dict={z:z0})
            z0 = sphere(args.batch,args.m)
            _,gc_,gn_,_, en2_ = sess.run([gtrain,gcost,gnorm, etrain_fake, enorm_fake], feed_dict={z:z0})
            ec2_ = -gc_
            ec+=ec_
            gc+=gc_
            en+=en_
            gn+=gn_
            ec2+=ec2_
            en2+=en2_
            t+=1.

        print 'epoch',i,'ecost_r',ec/t,'ecost_f',ec2/t, 'gcost',gc/t,'enorm_r',en/t,'enorm_f', en2/t, 'gnorm',gn/t



        z0 = sphere(args.batch,args.m)
        x0 = sess.run(gz, feed_dict={z:z0})
        x0 = np.clip(x0,0.,1.)*255.
  	rid = np.random.randint(nimg, size=10)
        x1 = d[rid]
        y0 = sess.run(gex, feed_dict={x:x1})
        y0 = np.clip(y0, 0., 1.)*255.
        x1 = np.clip(x1, 0., 1.)*255.
        img1 = np.concatenate((x0[0:10]).astype('uint8'), axis=1)
        img2 = np.concatenate(x1.astype('uint8'), axis=1)
        img3 = np.concatenate(y0.astype('uint8'), axis=1)
        theImage = cv2.resize(np.concatenate((img1, img2, img3), axis=0), (1000, 300))

        # theImage = cv2.resize(np.concatenate((x0[0:10]).astype('uint8'),axis=1),(1000,100))
        cv2.imshow(tag, theImage)

        if i%100 == 99:
            pngfname = "%s-%d.png" % (tag, i)
            modelname = "%s-%d.proto" % (tag, i)
            pathname = os.path.join(tag, pngfname)
            pathmodel = os.path.join(tag, modelname)
            cv2.imwrite(pathname, theImage);
            with open(os.devnull, 'w') as sys.stdout:
                graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout'])
            sys.stdout=sys.__stdout__
            tf.train.write_graph(graph, '.', pathmodel, as_text=False)
            
        cv2.waitKey(10)

        # write model, redirect stderr to supress annoying messages
        with open(os.devnull, 'w') as sys.stdout:
            graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout'])
        sys.stdout=sys.__stdout__
        tf.train.write_graph(graph, '.', args.model, as_text=False)

