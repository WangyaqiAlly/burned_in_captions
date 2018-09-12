#CUDA_VISIBLE_DEVICES='0' python ac-gan_tf.py
#hello
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
parser.add_argument('--m', help='latent space dimensionality', default=38, type=int)
parser.add_argument('--cat_dim', help='total categorical factors', default=10, type=int)
parser.add_argument('--con_dim', help='total continuous factors', default=2, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
parser.add_argument('--decay_steps', help='learning rate decay parameter', default=10000, type=int)
parser.add_argument('--decay_rate', help='learning rate decay parameter', default=0.95, type=float)
parser.add_argument('--batch', help='batch size', default=32, type=int)
parser.add_argument('--epochs', help='training epochs', default=10000, type=int)
parser.add_argument('--model', help='output model', default='model.proto')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--label_rate',help='fraction of sample with labels',default=0.5,type=float)
parser.add_argument('--data_fraction',help='fraction of data for training',default=1.0,type=float)
parser.add_argument('--img_rep',help='replication of digits shown',default=1,type=int)
args = parser.parse_args()
print args

with open('train-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    data = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    data = data/255.-0.5
    print (args.batch*int(float(data.shape[0])*args.data_fraction/args.batch))
    data = data[0:(args.batch*int(float(data.shape[0])*args.data_fraction/args.batch))]

with open('train-labels-idx1-ubyte','rb') as f:
    h = struct.unpack('>II',f.read(8))
    label = np.fromstring(f.read(), dtype=np.uint8).astype('int32')
    label = label[0:(args.batch*int(float(label.shape[0])*args.data_fraction/args.batch))]
    label = np.eye(10)[label]
    label_sel = np.random.uniform(0,1,label.shape[0])
    label_sel = (label_sel<args.label_rate).astype('float32')
#    label_sel = label_sel.reshape((-1,1))
#    label_sel2 = np.repeat(label_sel,10,axis=1)
#    label = np.dot(label_sel,label2)
    for isel in range(label.shape[0]):
        for jsel in range(label.shape[1]):
            label[isel][jsel] = label[isel][jsel]*label_sel[isel]


print 'data.shape',data.shape,'data.min()',data.min(),'data.max()',data.max(),'label.shape',label.shape
print 'fraction of image with labels:',np.sum(label)/data.shape[0]
def gather_cols(params, indices, name=None):
#    """Gather columns of a 2D tensor.

 #   Args:
 #       params: A 2D tensor.
#        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
#        name: A name for the operation (optional).

 #   Returns:
 #       A 2D Tensor. Has the same type as ``params``.
 #   """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],[-1, 1]) + indices, [-1])
    return tf.reshape(tf.gather(p_flat, i_flat),[p_shape[0], -1])

def enet(args,x,reuse=None):
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
     #   e = tf.concat([e,l],axis=1) ; print e
      #  e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
      #  e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
      #  e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
        e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
        cat = tf.layers.dense(inputs=e,units=10, activation=tf.nn.elu); print cat
        con = tf.layers.dense(inputs=e,units=2,  activation=tf.nn.sigmoid); print con
        e = tf.layers.dense(inputs=e, units=10, activation=tf.nn.elu) ; print e
        e = tf.layers.dense(inputs=e, units=1, activation=tf.sigmoid) ; print e
        e = tf.identity(e,name='eout') ; print e
    return e, cat, con


def gnet(args,z,reuse=None):
    print 'generator network, reuse', reuse
    with tf.variable_scope('gnet',reuse=reuse):
#        l = tf.layers.dense(inputs=z, units=100, activation=tf.nn.elu) ; print l
#        l = tf.layers.dense(inputs=l, units=100, activation=tf.nn.elu) ; print l
#        l = tf.layers.dense(inputs=l, units=100, activation=tf.nn.elu) ; print l
#        l = tf.layers.dense(inputs=l, units=100, activation=tf.nn.elu) ; print l
#        l = tf.layers.dense(inputs=l, units=10, activation=tf.nn.softmax) ; print l
#        l = tf.identity(l,name='lout') ; print l
#        l = tf.gather_cols(z,[0,1,2,3,4,5,6,7,8,9]) ;
        ind = tf.constant([0,1,2,3,4,5,6,7,8,9])
        #l = tf.transpose(tf.nn.embedding_lookup(tf.transpose(z),ind))
        l = tf.transpose(tf.nn.embedding_lookup(tf.transpose(z_cat),ind))
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
z_cat = tf.placeholder('float32', [None,args.cat_dim],name='z_cat') ; print z_cat
z_con = tf.placeholder('float32', [None,args.con_dim],name='z_con') ; print z_con
z_rand = tf.placeholder('float32', [None,args.m],name='z_rand') ; print z_rand
#z = tf.concat(axis=1, values=[tf.one_hot(z_cat,depth=args.cat_dim), z_con, z_rand])
z = tf.concat(axis=1, values=[z_cat, z_con, z_rand])
l = tf.placeholder('float32', [None,args.cat_dim],name='l') ; print l

ex,cat_real,_ = enet(args,x) # e(x)
gx,gl = gnet(args,z) # g(z)
egz,cat_fake,con_fake = enet(args,gx,reuse=True) # e(g(z))

# discriminate loss

eloss = -tf.reduce_mean(tf.log(ex) + tf.log(1.-egz))
# generate loss
gloss = -tf.reduce_mean(tf.log(egz))
# AUXILIARY categorical factor loss

closs = 0.5* tf.nn.softmax_cross_entropy_with_logits(labels=z_cat,logits=cat_fake) \
       +1/args.label_rate *0.5* tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=l,logits=cat_real),tf.reduce_sum(l,axis=1)) 
# AUXILIARY continuous factor loss
conloss = tf.losses.mean_squared_error(labels=z_con,predictions=con_fake)


global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(args.lr, global_step, args.decay_steps, args.decay_rate, staircase=True)

eopt = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
#egrads = eopt.compute_gradients(eloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
egrads = eopt.compute_gradients(eloss + closs + conloss ,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'enet'))
etrain = eopt.apply_gradients(egrads,global_step=global_step)
enorm = tf.global_norm([i[0] for i in egrads])

gopt = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
#ggrads = gopt.compute_gradients(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet'))
ggrads = gopt.compute_gradients(gloss + closs + conloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gnet'))
gtrain = gopt.apply_gradients(ggrads,global_step=global_step)
gnorm = tf.global_norm([i[0] for i in ggrads])

init = tf.global_variables_initializer()

# def creat_z(args):
#     z_cat = tf.multinomial(tf.ones((args.batch, args.cat_dim), dtype='int32') / args.cat_dim, 1).squeeze().cast(tf.int32)
#   # continuous latent variable
#     z_con = tf.random_normal((args.batch, args.con_dim))
#   # random latent variable dimension
#     z_rand = tf.random_normal((args.batch, args.m))
#   # latent variable
#     #z = tf.concat(axis=1, values=[z_cat.one_hot(depth=args.cat_dim), z_con, z_rand])
#     return z_cat, z_con, z_rand
    
def creat_z_cat(args):
    z_cat = np.zeros((args.batch,args.cat_dim),dtype='float32')
    for i in xrange(args.batch):
        ind=np.where(np.random.multinomial(1, np.ones((args.cat_dim)) / args.cat_dim)==1)[0][0]
        np.put(z_cat[i],ind,1)
    return z_cat

def creat_show_z_cat(arg,numz):
    z_cat = np.zeros((numz,args.cat_dim),dtype='float32')
    for i in xrange(numz):
      #  ind=np.where(np.random.multinomial(1, np.ones((args.cat_dim)) / args.cat_dim)==1)[0][0]
        ind = i % 10
        np.put(z_cat[i],ind,1)
    return z_cat

def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

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
            #_,el_,en_ = sess.run([etrain,eloss,enorm],feed_dict={x:data[j:j+args.batch],l:label[j:j+args.batch],z:np.random.randn(args.batch,args.m)})

            _,el_,en_ = sess.run([etrain,eloss,enorm],feed_dict={x:data[j:j+args.batch],
                                                                 l:label[j:j+args.batch],
                                                                 z_cat:creat_z_cat(args),
                                                                 z_con : np.random.normal(size=(args.batch, args.con_dim)),
                                                                 z_rand : np.random.normal(size=(args.batch, args.m))})
            et+=1.
            elb+=el_
            enb+=en_
            #sys.stdout.write('\r')
            #sys.stdout.write("%d%%" % int(j/data.shape[0]*100)
            #sys.stdout.flush()
            update_progress(  int((j+1.0)/(data.shape[0]*1.0)*1000+1)/1000.0 )

            #_,gl_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={z:np.random.randn(args.batch,args.m)})
            _,gl_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={x:data[j:j+args.batch],
                                                                  l:label[j:j+args.batch],
                                                                  z_cat:creat_z_cat(args),
                                                                  z_con : np.random.normal(size=(args.batch, args.con_dim)),
                                                                  z_rand : np.random.normal(size=(args.batch, args.m))})
            gt+=1.
            glb+=gl_
            gnb+=gn_
        
            #_,gl_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={z:np.random.randn(args.batch,args.m)})
            _,gl_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={x:data[j:j+args.batch],
                                                                  l:label[j:j+args.batch],
                                                                  z_cat:creat_z_cat(args),
                                                                  z_con : np.random.normal(size=(args.batch, args.con_dim)),
                                                                  z_rand : np.random.normal(size=(args.batch, args.m))})
            gt+=1.
            glb+=gl_
            gnb+=gn_
        print '\n epoch {:6d} eloss {:12.8f} gloss {:12.8f} egrad {:12.8f} ggrad {:12.8f} learning_rate {:12.8f}'.format(i,elb/et,glb/gt,enb/et,gnb/gt,sess.run(learning_rate))

        #x0,l0 = sess.run([gx,gl],feed_dict={z:np.random.randn(args.batch,args.m)})

        temp_z_cat = creat_show_z_cat(args,10*args.img_rep)
        #print temp_z_cat
        x0,l0,cat_enet = sess.run([gx,gl,cat_fake],feed_dict={#x:data[j:j+args.batch],
                                           # l:label[j:j+args.batch],
                                            z_cat:temp_z_cat,
                                            z_con : np.random.normal(size=(10*args.img_rep, args.con_dim)),
                                            z_rand : np.random.normal(size=(10*args.img_rep, args.m))})
        
       # print loo
        x0 = np.clip(x0+0.5,0.,1.)*255.
        x02 = np.concatenate((x0[0:10]).astype('uint8'),axis=1)
        #xt = np.concatenate((x0[0:10]).astype('uint8'),axis=1)
       # x03 = np.concatenate((x02,xt),axis=0)
      #  print x03.shape

       
        for imjj in range(args.img_rep-1):
            xt = np.concatenate((x0[(imjj*10+10):(imjj*10+20)]).astype('uint8'),axis=1)
            x02 = np.concatenate((x02,xt),axis=0)
            

        
        #print x02.shape 
        #print l0


        img = cv2.cvtColor(cv2.resize(x02,(1000,1000)),cv2.COLOR_GRAY2RGB)
        for k in range(10):
            for k2 in range(args.img_rep):
                cv2.putText(img, "{:d}".format(np.argmax(l0[k+k2*10])),(k*100,k2*100+25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
                cv2.putText(img, "{:d}".format(np.argmax(cat_enet[k+k2*10])),(k*100,k2*100+95),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(10)

        # write model, redirect stderr to supress annoying messages
        with open(os.devnull, 'w') as sys.stdout:
            graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout'])
        sys.stdout=sys.__stdout__
        tf.train.write_graph(graph, '.', args.model, as_text=False)


