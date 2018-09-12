#CUDA_VISIBLE_DEVICES='0' python ac-gan_cifar_orig.py
#hello
import argparse
import struct
import time
import sys
import os,glob
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
from time import gmtime, strftime
#import h5py

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=110, type=int)
parser.add_argument('--cat_dim', help='total categorical factors', default=10, type=int)
parser.add_argument('--con_dim', help='total continuous factors', default=2, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
parser.add_argument('--decay_steps', help='learning rate decay parameter', default=10000, type=int)
parser.add_argument('--decay_rate', help='learning rate decay parameter', default=0.95, type=float)
parser.add_argument('--batch', help='batch size', default=100, type=int)
parser.add_argument('--epochs', help='training epochs', default=10000, type=int)
parser.add_argument('--model', help='output model', default='model.proto')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--label_rate',help='fraction of sample with labels',default=1.0,type=float)
parser.add_argument('--data_fraction',help='fraction of data for training',default=1.0,type=float)
parser.add_argument('--img_rep',help='replication of digits shown',default=10,type=int)
parser.add_argument('--img_rep_save',help='replication of digits shown',default=100,type=int)
parser.add_argument('--dataFolder', help='data folder', default='./cifar-10-batches-py/')
parser.add_argument('--save_img_freq',help='frequency to same sampled images to save',default=10,type=float)
args = parser.parse_args()
print args

import cPickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
dataFolder = args.dataFolder
fileNames = os.listdir(dataFolder)
data = []
label = []
myFlag = True
for f in fileNames:
  if 'data_batch' in f:
    tempDict = unpickle(dataFolder+f)
    tempArr = tempDict['data']
    tempLabel = tempDict['labels']
    for i in range(len(tempLabel)):
      if tempLabel[i]!=10000:
        #d.append(tempArr[i])
        im = np.reshape(tempArr[i],(32,32,3),order='F')
        rows,cols,_ = im.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        im = cv2.warpAffine(im,M,(cols,rows))
        #gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #print gray_image.shape
        tempOneHot = np.eye(10)[tempLabel[i]]
        #tempOneHot[tempLabel[i]]=1

        #tempOneHot
        data.append(im)
        label.append(tempOneHot)
        '''cv2.imshow('myimg',cv2.resize(im,(100,100)))
        k = cv2.waitKey(0)
              if k==1114083: # ctrl-c to exit
          break
        elif k==115:
                      d.append(gray_image)
          if len(d)==10:
            myFlag = False'''
      if myFlag==False:
        break
    if myFlag==False:
      break
        #cv2.imshow('myimg',cv2.resize(im,(100,100)))
        #cv2.waitKey(1000)
    #d.append(tempArr.reshape((10000,32,32,3)).astype('float32'))
#d = np.concatenate((d[0],d[1],d[2],d[3],d[4]),axis=0)
numTrainSamples = len(data)
data = np.array(data[0:numTrainSamples]).reshape(numTrainSamples,32,32,3).astype('float32')
#d.reshape((50000,32,32,3)).astype('float32')
data = data/255.
data = data-0.5
data = data[0:(int(data.shape[0]/args.batch)*args.batch)]
data = data[0:(args.batch*int(float(data.shape[0])*args.data_fraction/args.batch))]
label = np.array(label[0:numTrainSamples]).reshape(numTrainSamples,10).astype('float32')
label = label[0:data.shape[0]]
label_sel = np.random.uniform(0,1,label.shape[0])
label_sel = (label_sel<args.label_rate).astype('float32')
#    label_sel = label_sel.reshape((-1,1))
#    label_sel2 = np.repeat(label_sel,10,axis=1)
#    label = np.dot(label_sel,label2)
for isel in range(label.shape[0]):
    for jsel in range(label.shape[1]):
        label[isel][jsel] = label[isel][jsel]*label_sel[isel]
print label.shape
label_name=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
save_dir = "./asset/"+'_labelrate='+str(args.label_rate)+'_datafraction='+str(args.data_fraction)+'_'+strftime("%Y-%m-%d_%H-%M-%S", gmtime())+"/"
model_dir = save_dir+'model/'
sample_dir = save_dir+'samples/'
try:
    os.stat(model_dir)
except:
    os.makedirs(model_dir)
try:
    os.stat(sample_dir)
except:
    os.makedirs(sample_dir)

import shutil
os.chdir(os.getcwd())
for file in glob.glob("*.py"):
    shutil.copyfile(file,save_dir+file)
f = open(save_dir+'args.txt', 'w')
print >>f, args
f.close()

#with open('train-images-idx3-ubyte','rb') as f:
#    h = struct.unpack('>IIII',f.read(16))
#    d = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
#    d = d/255.
#print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()


# headline = np.zeros((10,320,3),dtype='uint8')
# x0 = np.clip(x0+0.5,0.,1.)*255.
# #x02 = np.concatenate((x0[0:10]).astype('uint8'),axis=1)
# x02 = np.concatenate(((data[0:10]+0.5)*255).astype('uint8'),axis=1)
# #print x02.shape
# #print headline.shape
# x02 = np.concatenate((headline,x02),axis=0)

        

       
# for imjj in range(args.img_rep-1):
#     xt = np.concatenate((data[(imjj*10+10):(imjj*10+20)]*255+128).astype('uint8'),axis=1)
#     x02 = np.concatenate((x02,xt),axis=0)
            

        
      
        
#     #img = cv2.cvtColor(cv2.resize(x02,(1000,1000)),cv2.COLOR_GRAY2RGB)
#     img = cv2.resize(x02,(960,96*args.img_rep+30))
#     # cv2.putText(img,"cat",(10,10),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
#     headlineTexts = ['0:airplane','1:auto','2:bird','3:cat','4:deer','5:dog','6:frog','7:horse','8:ship','9:truck']
#     for k in range(10):
#         cv2.putText(img,headlineTexts[k],(1+96*k,18),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
#         for k2 in range(args.img_rep):
#             #cv2.putText(img, "{:d}".format(np.argmax(l0[k+k2*10])),(k*100,k2*100+25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
#             cv2.putText(img, "{:d}".format(np.argmax(label[k+k2*10])),(k*96,k2*96+120),cv2.FONT_HERSHEY_SIMPLEX,0.90,(0,255,0),2)
#       #  cv2.imshow('img',img)
#         cv2.waitKey(10)




print 'data.shape',data.shape,'data.min()',data.min(),'data.max()',data.max(),'label.shape',label.shape
print 'fraction of image with labels:',np.sum(label)/data.shape[0]
def save_img(img,fig_name,save_dir):
    col = np.ceil(np.sqrt(len(img))).astype('int32')
    row = col
    img = (img + 0.5)*255
    img = np.clip(img,0,255)
    x02 = np.concatenate((img[0:row]).astype('uint8'),axis=1)
    for imjj in range(col-1):
        xt = np.concatenate((img[(imjj*10+10):(imjj*10+20)]).astype('uint8'),axis=1)
        #xt = np.concatenate((data[(imjj*10+10):(imjj*10+20)]*255+128).astype('uint8'),axis=1)
        x02 = np.concatenate((x02,xt),axis=0)
    x03 = x02[...,[2,1,0]]
    imgname = save_dir+"/"+fig_name
    cv2.imwrite(imgname,x03)
def pullaway_loss(args,embeddings):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
    pt_loss = (tf.reduce_sum(similarity) - args.batch) / (args.batch * (args.batch - 1))
    return pt_loss

def creat_z_cat_nonrand(args, num_sample, digit):
    z_cat = np.zeros((num_sample,args.cat_dim),dtype='float32')
    z_cat[:,digit] = 1
    return z_cat

def leaky_relu(x):
    return tf.where(tf.greater(x, 0), x, 0.2 * x)


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

def enet(args,x,keep_rate,phase,reuse=None):
    print 'encoder network, reuse',reuse
    with tf.variable_scope('enet',reuse=reuse):
        e = tf.layers.conv2d(inputs=x, filters=16, kernel_size=3, strides=2,activation=None, padding='same') ; print e
        e = leaky_relu(e)
        e = tf.nn.dropout(e,keep_rate)
        e = tf.layers.conv2d(inputs=e, filters=32, kernel_size=3, strides=1,activation=None, padding='same') ; print e
        e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn')
        e = leaky_relu(e)
        
        e = tf.nn.dropout(e,keep_rate)
        
        #e = tf.image.resize_bilinear(images=e,size=[16,16]) ; print e
        e = tf.layers.conv2d(inputs=e, filters=64, kernel_size=3, strides=2,activation=None, padding='same') ; print e
        e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn2')
        e = leaky_relu(e)
        e = tf.nn.dropout(e,keep_rate)
        
        e = tf.layers.conv2d(inputs=e, filters=128, kernel_size=3, strides=1,activation=None, padding='same') ; print e
        e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn3')
        e = leaky_relu(e)
        e = tf.nn.dropout(e,keep_rate)
        
        #e = tf.image.resize_bilinear(images=e,size=[8,8]) ; print e
        e = tf.layers.conv2d(inputs=e, filters=256, kernel_size=3, strides=2,activation=None, padding='same'); print e
        e = tf.contrib.layers.batch_norm(e,center=True,scale=True,is_training=phase,scope='bn4')
        e = leaky_relu(e)
        e = tf.nn.dropout(e,keep_rate)
        
        e = tf.layers.conv2d(inputs=e, filters=512, kernel_size=3, strides=1,activation=None, padding='same') ; print e
        e = tf.contrib.layers.flatten(e) ; print e
        #e = tf.layers.dense(inputs=e, units=11, activation=tf.nn.elu) ; print e
        #e = tf.layers.dense(inputs=e, units=l.shape[1], activation=tf.sigmoid) ; print e
       # e = tf.concat([e,l],axis=1) ; print e
      #  e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
      #  e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
      #  e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
      #  e = tf.layers.dense(inputs=e, units=100, activation=tf.nn.elu) ; print e
        cat = tf.layers.dense(inputs=e,units=10, activation=tf.nn.elu); print cat
        con = tf.layers.dense(inputs=e,units=2,  activation=tf.nn.sigmoid); print con
        e = tf.layers.dense(inputs=e, units=10, activation=tf.nn.elu) ; print e
        e = tf.layers.dense(inputs=e, units=1, activation=tf.sigmoid) ; print e
        e = tf.identity(e,name='eout') ; print e
    return e, cat, con


def gnet(args,z,phase,reuse=None):
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
        g = tf.layers.dense(inputs=z, units=384, activation=tf.nn.relu) ; print g
        g = tf.reshape(g,[-1,4,4,24]) ; print g
        g = tf.layers.conv2d_transpose(inputs=g, filters=192, kernel_size=5, strides=2,activation=None, padding='same') ; print g
        g = tf.contrib.layers.batch_norm(g,center=True,scale=True,is_training=phase,scope='gbn1')
        g = tf.nn.relu(g)
        g = tf.layers.conv2d_transpose(inputs=g, filters=96, kernel_size=5, strides=2,activation=None, padding='same') ; print g
        g = tf.contrib.layers.batch_norm(g,center=True,scale=True,is_training=phase,scope='gbn2')
        g = tf.nn.relu(g)
        #g = tf.image.resize_bilinear(images=g,size=[16,16]) ; print g
        g = tf.layers.conv2d_transpose(inputs=g, filters=3, kernel_size=5, strides=2,activation=tf.nn.tanh, padding='same') ; print g
        #g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        #g = tf.image.resize_bilinear(images=g,size=[32,32]) ; print g
        #g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        #g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        #g = tf.layers.conv2d(inputs=g, filters=3, kernel_size=3, strides=1,activation=None, padding='same') ; print g
        #g = tf.identity(g,name='gout') ; print g
    return g,l

#print datat.shape, labelt.shape
phase = tf.placeholder(tf.bool,name='phase')
x = tf.placeholder('float32', [None,32,32,3],name='x') ; print x
z_cat = tf.placeholder('float32', [None,args.cat_dim],name='z_cat') ; print z_cat
z_con = tf.placeholder('float32', [None,args.con_dim],name='z_con') ; print z_con
z_rand = tf.placeholder('float32', [None,args.m],name='z_rand') ; print z_rand
#z = tf.concat(axis=1, values=[tf.one_hot(z_cat,depth=args.cat_dim), z_con, z_rand])
z = tf.concat(axis=1, values=[z_cat, z_con, z_rand])
l = tf.placeholder('float32', [None,args.cat_dim],name='l') ; print l

ex,cat_real,_ = enet(args,x,0.5,phase) # e(x)
gx,gl = gnet(args,z,phase) # g(z)
egz,cat_fake,con_fake = enet(args,gx,0.5,phase,reuse=True) # e(g(z))
egz2,cat_fake_full,con_fake2 = enet(args,gx,0.5,phase,reuse=True)

# discriminate loss

eloss = -tf.reduce_mean(tf.log(tf.clip_by_value(ex,0.00000001,sys.float_info.max)) + tf.log(tf.clip_by_value((1.-egz),0.00000001,sys.float_info.max)))
# generate loss
gloss = -tf.reduce_mean(tf.log(tf.clip_by_value(egz,0.00000001,sys.float_info.max)))#+pullaway_loss(args,egz)
# AUXILIARY categorical factor loss

closs = tf.reduce_mean(0.5* tf.nn.softmax_cross_entropy_with_logits(labels=z_cat,logits=cat_fake) \
       +1/args.label_rate *0.5* tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=l,logits=cat_real),tf.reduce_sum(l,axis=1)) )
# AUXILIARY continuous factor loss
#import pdb; pdb.set_trace()
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
    barLength = 50 # Modify this to change the length of the progress bar
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
        conlb=0.
        clb=0.

        for j in range(0,data.shape[0],args.batch):
            #_,el_,en_ = sess.run([etrain,eloss,enorm],feed_dict={x:data[j:j+args.batch],l:label[j:j+args.batch],z:np.random.randn(args.batch,args.m)})

            _,el_,en_ = sess.run([etrain,eloss,enorm],feed_dict={x:data[j:j+args.batch],
                                                                 l:label[j:j+args.batch],
                                                                 phase:True,
                                                                 z_cat:creat_z_cat(args),
                                                                 z_con : np.random.normal(size=(args.batch, args.con_dim)),
                                                                 z_rand : np.random.normal(size=(args.batch, args.m))})
            et+=1.
            elb+=el_
            enb+=en_
            #sys.stdout.write('\r')
            #sys.stdout.write("%d%%" % int(j/data.shape[0]*100)
            #sys.stdout.flush()
            update_progress(  int((j+args.batch)/(data.shape[0]*1.0)*1000)/1000.0 )

            #_,gl_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={z:np.random.randn(args.batch,args.m)})
            _,gl_,gn_,conloss_,closs_  = sess.run([gtrain,gloss,gnorm,conloss,closs], feed_dict={x:data[j:j+args.batch],
                                                                  l:label[j:j+args.batch],
                                                                  phase:True,
                                                                  z_cat:creat_z_cat(args),
                                                                  z_con : np.random.normal(size=(args.batch, args.con_dim)),
                                                                  z_rand : np.random.normal(size=(args.batch, args.m))})
            gt+=1.
            glb+=gl_
            gnb+=gn_
            conlb+=conloss_
            clb+=closs_
        
            #_,gl_,gn_ = sess.run([gtrain,gloss,gnorm], feed_dict={z:np.random.randn(args.batch,args.m)})
            _,gl_,gn_,conloss_,closs_  = sess.run([gtrain,gloss,gnorm,conloss,closs], feed_dict={x:data[j:j+args.batch],
                                                                  l:label[j:j+args.batch],
                                                                  phase:True,
                                                                  z_cat:creat_z_cat(args),
                                                                  z_con : np.random.normal(size=(args.batch, args.con_dim)),
                                                                  z_rand : np.random.normal(size=(args.batch, args.m))})
            gt+=1.
            glb+=gl_
            gnb+=gn_
            conlb+=conloss_
            clb+=closs_
        print '\n epoch {:6d} eloss {:6.4f} gloss {:6.4f} conloss {:6.4f} closs {:6.4f} egrad {:6.4f} ggrad {:6.4f} learning_rate {:12.4f}'.format(i,elb/et,glb/gt,conlb/gt,clb/gt,enb/et,gnb/gt,sess.run(learning_rate))

        #x0,l0 = sess.run([gx,gl],feed_dict={z:np.random.randn(args.batch,args.m)})

        temp_z_cat = creat_show_z_cat(args,10*args.img_rep)
        #print temp_z_cat
        x0,l0,cat_enet = sess.run([gx,gl,cat_fake_full],feed_dict={#x:data[j:j+args.batch],
                                           # l:label[j:j+args.batch],
                                            phase: True,
                                            z_cat:temp_z_cat,
                                            z_con : np.random.normal(size=(10*args.img_rep, args.con_dim)),
                                            z_rand : np.random.normal(size=(10*args.img_rep, args.m))})
        
       # print loo
        headline = np.zeros((10,320,3),dtype='uint8')
        x0 = np.clip(x0+0.5,0.,1.)*255.
       #print   'x0 shape',x0.shape 
        x02 = np.concatenate((x0[0:10]).astype('uint8'),axis=1)
        #x02 = np.concatenate(((data[0:10]+0.5)*255).astype('uint8'),axis=1)
        #print x02.shape
        #print headline.shape
        x02 = np.concatenate((headline,x02),axis=0)

        

       
        for imjj in range(args.img_rep-1):
            xt = np.concatenate((x0[(imjj*10+10):(imjj*10+20)]).astype('uint8'),axis=1)
            #xt = np.concatenate((data[(imjj*10+10):(imjj*10+20)]*255+128).astype('uint8'),axis=1)

            x02 = np.concatenate((x02,xt),axis=0)
            

        
      
        x03 = x02[...,[2,1,0]]   #RGB to BGR 
        #img = cv2.cvtColor(cv2.resize(x02,(1000,1000)),cv2.COLOR_GRAY2RGB)
        img = cv2.resize(x03,(960,96*args.img_rep+30))

        # cv2.putText(img,"cat",(10,10),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
        headlineTexts = ['0:airplane','1:auto','2:bird','3:cat','4:deer','5:dog','6:frog','7:horse','8:ship','9:truck']
        for k in range(10):
            cv2.putText(img,headlineTexts[k],(1+96*k,18),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            for k2 in range(args.img_rep):
                #cv2.putText(img, "{:d}".format(np.argmax(l0[k+k2*10])),(k*100,k2*100+25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
                cv2.putText(img, "{:d}".format(np.argmax(cat_enet[k+k2*10])),(k*96,k2*96+120),cv2.FONT_HERSHEY_SIMPLEX,0.90,(0,255,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(10)
        if i%args.save_img_freq ==0:
            for d in xrange(10):
                temp_z_cat = creat_z_cat_nonrand(args,num_sample=args.img_rep_save,digit=d)

                x0,cat_enet = sess.run([gx,cat_fake_full],feed_dict={z_cat: temp_z_cat,
                                                                      phase:True,
                                                                      z_con : np.random.normal(size=(args.img_rep_save, args.con_dim)),
                                                                      z_rand : np.random.normal(size=(args.img_rep_save, args.m))})

                save_img(x0,'epoch%d_%s.jpg' %(i,label_name[d]),sample_dir)

        # write model, redirect stderr to supress annoying messages
       # with open(os.devnull, 'w') as sys.stdout:
        #    graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['enet/eout','gnet/gout'])
        #sys.stdout=sys.__stdout__
        #tf.train.write_graph(graph, '.', args.model, as_text=False)


