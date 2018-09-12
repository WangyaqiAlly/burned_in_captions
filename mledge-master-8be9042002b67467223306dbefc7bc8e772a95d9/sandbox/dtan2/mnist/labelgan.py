# CUDA_VISIBLE_DEVICES='1' python gan.py
import argparse
import time
import sys
import os
import numpy as np
import mnist

print 'numpy ' + np.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print 'tensorflow ' + tf.__version__
import cv2
print 'cv2 ' + cv2.__version__

TINY = 1e-9
#TINY = 0

tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]

default_model = tag + '.proto'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=38, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lrd', help='discriminator learning rate', default=0.0002, type=float)
parser.add_argument('--lrg', help='generator learning rate', default=0.0001, type=float)
parser.add_argument('--batch', help='batch size', default=1024, type=int)
parser.add_argument('--epochs', help='training epochs', default=1000000, type=int)
parser.add_argument('--model', help='output model', default=default_model)
parser.add_argument('--step', help='step to start training', default=0, type=int)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

if not os.path.exists(tag):
    os.makedirs(tag)

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
        e = tf.layers.dense(inputs=e, units=11, activation=None); print e
        e = tf.identity(e,name='eout'); print e
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

def genZ(batch, d, override=False):
    z0 = np.random.randn(batch, d)
    return z0

def gen_onehot_label(lab):
    batch = lab.shape[0]
    real = np.zeros((batch, 11))
    real[np.arange(batch), lab] = 1
    fake = np.zeros((batch, 11))
    fake[np.arange(batch), [10]*batch] = 1
    return real, fake

step = args.step
sess = tf.Session()

x = tf.placeholder('float32', [None,28,28,1],name='x') ; print x
z = tf.placeholder('float32', [None,args.m],name='z') ; print z
y = tf.placeholder('float32', [None, 11], name='y'); print y
fake_label = tf.placeholder('float32', [None,11], name='fake_label'); print fake_label

mut_real = enet(args,x) # e(x)
gz = gnet(args,z) # g(z)
mut_fake = enet(args,gz,reuse=True) # e(g(z))

sm_fake = tf.nn.softmax(mut_fake)

#qloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=z[:,0:10], logits=mut_fake))
#eloss = -tf.reduce_mean(tf.log(ex+TINY) + tf.log(1. - egz + TINY))
eloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=mut_real) + tf.nn.softmax_cross_entropy_with_logits(labels=fake_label,logits=mut_fake))
gloss = -tf.reduce_mean(tf.log(1. - sm_fake[:,10] + TINY)); print sm_fake

eopt = tf.train.AdamOptimizer(learning_rate=args.lrd, beta1=0.9)
egrads = eopt.compute_gradients(eloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
etrain = eopt.apply_gradients(egrads)
enorm = tf.global_norm([i[0] for i in egrads])

gopt = tf.train.AdamOptimizer(learning_rate=args.lrg, beta1=0.9)
ggrads = gopt.compute_gradients(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet'))
gtrain = gopt.apply_gradients(ggrads)
gnorm = tf.global_norm([i[0] for i in ggrads])

saver = tf.train.Saver()

if step == 0:
    sess.run(tf.global_variables_initializer())
else:
    model_path = os.path.join(tag, 'model-' + str(step))
    saver.restore(sess, model_path)

data = mnist.MnistDataset()
nBatches = data.get_batch_in_epoch(args.batch)

if 1==1:
    for i in range(args.epochs):
        el=0.
        gl=0.
        en=0.
        gn=0.
        t=0.

        thelist = []
        for b in range(0,nBatches):
            j = b*args.batch
            z0 = genZ(args.batch, args.m)
            d,lab = data.next_batch(args.batch)
            real,fake = gen_onehot_label(lab)
            _,el_,en_ = sess.run([etrain,eloss,enorm],feed_dict={x:d,z:z0,y:real,fake_label:fake})

            for bias in range(3):
                z0 = genZ(args.batch, args.m)
                _,gl_,gn_,y0 = sess.run([gtrain,gloss,gnorm,mut_fake], feed_dict={z:z0})
                #labs = sess.run(tf.argmax(y0, axis=1))
                #hist = np.bincount(labs,minlength=11)
                #fake_ratio = float(hist[10]*100)/args.batch
                #thelist.append("{0:.2f}".format(fake_ratio))
            el+=el_
            gl+=gl_
            en+=en_
            gn+=gn_
            t+=1.

            step += 1
            if (step % 50) == 0:
                savepath = saver.save(sess, os.path.join(tag, 'model'), global_step=step, write_meta_graph=False)
                print 'saving ',savepath

        #print 'fake_ratio', thelist
        print 'epoch',i,'eloss',el/t,'gloss',gl/t,'enorm',en/t,'gnorm',gn/t

        z0 = genZ(10, args.m, override=True)
        x0,y0 = sess.run([gz, mut_fake], feed_dict={z:z0})
        labs,labs2 = sess.run([tf.argmax(y0, axis=1),tf.argmax(y0[:,0:10], axis=1)])
        x0 = np.clip(x0,0.,1.)*255.

        theImage = cv2.resize(np.concatenate((x0).astype('uint8'),axis=1),(1000,100))
        for m in range(10):
            ann = str(labs[m])
            if labs[m] == 10:
                ann = '(' + str(labs2[m]) + ')'
            cv2.putText(theImage, ann, (m*100+5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
        cv2.imshow(tag, theImage)
        cv2.waitKey(10)

        if i%20 == 19:
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
