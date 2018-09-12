# CUDA_VISIBLE_DEVICES='0' python labelgan.py
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
parser.add_argument('--m', help='latent space dimensionality', default=10, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
parser.add_argument('--decay_steps', help='learning rate decay parameter', default=10000, type=int)
parser.add_argument('--decay_rate', help='learning rate decay parameter', default=0.95, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=1000000, type=int)
parser.add_argument('--model', help='output model', default='model.proto')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

with open('train-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    data = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    data = data/255.-0.5

with open('train-labels-idx1-ubyte','rb') as f:
    h = struct.unpack('>II',f.read(8))
    label = np.fromstring(f.read(), dtype=np.uint8).astype('int32')
    label = np.eye(10)[label]

print 'data.shape',data.shape,'data.min()',data.min(),'data.max()',data.max(),'label.shape',label.shape

def enet(args,x,l,reuse=None):
    print 'encoder network, reuse',reuse
    with tf.variable_scope('enet',reuse=reuse):
        e = tf.layers.conv2d(inputs=x, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.layers.conv2d(inputs=e, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.image.resize_bilinear(images=e,size=[14,14]) ; print e
        e = tf.layers.conv2d(inputs=e, filters=2*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.layers.conv2d(inputs=e, filters=2*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.image.resize_bilinear(images=e,size=[7,7]) ; print e
        e = tf.layers.conv2d(inputs=e, filters=4*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.layers.conv2d(inputs=e, filters=4*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print e
        e = tf.contrib.layers.flatten(e) ; print e
        e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
        e = tf.layers.dense(inputs=e, units=l.shape[1], activation=tf.nn.elu) ; print e
        e = tf.concat([e,l],axis=1) ; print e
        e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
        e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
        e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
        e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
        e = tf.layers.dense(inputs=e, units=10, activation=tf.nn.elu) ; print e
        e = tf.layers.dense(inputs=e, units=1, activation=tf.sigmoid) ; print e
        e = tf.identity(e,name='eout') ; print e
    return e

def gnet(args,z,reuse=None):
    print 'generator network, reuse', reuse
    with tf.variable_scope('gnet',reuse=reuse):
        l = tf.layers.dense(inputs=z, units=100, activation=tf.nn.elu) ; print l
        l = tf.layers.dense(inputs=l, units=100, activation=tf.nn.elu) ; print l
        l = tf.layers.dense(inputs=l, units=100, activation=tf.nn.elu) ; print l
        l = tf.layers.dense(inputs=l, units=100, activation=tf.nn.elu) ; print l
        l = tf.layers.dense(inputs=l, units=10, activation=tf.nn.softmax) ; print l
        l = tf.identity(l,name='lout') ; print l
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
        g = tf.layers.conv2d(inputs=g, filters=1, kernel_size=3, strides=1,activation=None, padding='same') ; print g
        g = tf.identity(g,name='gout') ; print g
    return g,l

x = tf.placeholder('float32', [None,28,28,1],name='x') ; print x
z = tf.placeholder('float32', [None,args.m],name='z') ; print z
l = tf.placeholder('float32', [None,10],name='l') ; print l

ex = enet(args,x,l) # e(x)
gx,gl = gnet(args,z) # g(z)
egz = enet(args,gx,gl,reuse=True) # e(g(z))

eloss = -tf.reduce_mean(tf.log(ex) + tf.log(1.-egz))
gloss = -tf.reduce_mean(tf.log(egz))

global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(args.lr, global_step, args.decay_steps, args.decay_rate, staircase=True)

eopt = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
egrads = eopt.compute_gradients(eloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
etrain = eopt.apply_gradients(egrads,global_step=global_step)
enorm = tf.global_norm([i[0] for i in egrads])

gopt = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
ggrads = gopt.compute_gradients(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet'))
gtrain = gopt.apply_gradients(ggrads,global_step=global_step)
gnorm = tf.global_norm([i[0] for i in ggrads])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        rng_state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(rng_state)
        np.random.shuffle(label)

        elb=0.
        glb=0.
        enb=0.
        gnb=0.
        et=0.
        gt=0.
        for j in range(0,data.shape[0],args.batch):
            _,el_,en_ = sess.run([etrain,eloss,enorm],feed_dict={x:data[j:j+args.batch],l:label[j:j+args.batch],z:np.random.randn(args.batch,args.m)})
            et+=1.
            elb+=el_
            enb+=en_

            _,gl_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={z:np.random.randn(args.batch,args.m)})
            gt+=1.
            glb+=gl_
            gnb+=gn_
        
            _,gl_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={z:np.random.randn(args.batch,args.m)})
            gt+=1.
            glb+=gl_
            gnb+=gn_
        
        print 'epoch {:6d} eloss {:12.8f} gloss {:12.8f} egrad {:12.8f} ggrad {:12.8f} learning_rate {:12.8f}'.format(i,elb/et,glb/gt,enb/et,gnb/gt,sess.run(learning_rate))

        x0,l0 = sess.run([gx,gl],feed_dict={z:np.random.randn(args.batch,args.m)})
        x0 = np.clip(x0+0.5,0.,1.)*255.
        img = cv2.cvtColor(cv2.resize(np.concatenate((x0[0:10]).astype('uint8'),axis=1),(1000,100)),cv2.COLOR_GRAY2RGB)
        for k in range(10):
            cv2.putText(img, "{:d}".format(np.argmax(l0[k])),(k*100,25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
        cv2.imshow('img',img)
        cv2.waitKey(10)

        # write model, redirect stderr to supress annoying messages
        with open(os.devnull, 'w') as sys.stdout:
            graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout','gnet/lout'])
        sys.stdout=sys.__stdout__
        tf.train.write_graph(graph, '.', args.model, as_text=False)
