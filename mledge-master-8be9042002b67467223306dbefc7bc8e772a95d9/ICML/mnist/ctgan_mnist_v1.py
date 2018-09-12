'''
    train_ctgan_mnist.py

'''

import os
import sys
import time
import argparse

import cv2
import struct
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images



'Fixed parameters'
Factor_M = 0.0    # factor M
LAMBDA   = 10. # Gradient penalty lambda hyperparameter
LAMBDA_2 = 2.0    # weight factor
DIM = 64 	  # Model dimensionality
CRITIC_ITERS = 5  # For WGAN and WGAN-GP, number of critic iters per gen iter
OUTPUT_DIM = 784  # Number of pixels in MNIST (28*28)

load_dir = None

lib.print_model_settings(locals().copy())

def load_dataset(datadir, tag): 

    if tag == 'train': 
        image_file = '%s/train-images-idx3-ubyte' % datadir
        label_file = '%s/train-labels-idx1-ubyte' % datadir
    else: 
        image_file = '%s/t10k-images-idx3-ubyte' % datadir
        label_file = '%s/t10k-labels-idx1-ubyte' % datadir

    with open(image_file,'rb') as f:
        h = struct.unpack('>IIII',f.read(16))
        d = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
        d = d/255.

    with open(label_file,'rb') as f:
        h = struct.unpack('>II',f.read(8))
        l = np.fromstring(f.read(), dtype=np.uint8).astype('int32')

    print '%s: ' % tag, 'd.shape',d.shape,'l.shape',l.shape,
    return d, l

def rshuffle(d, l):
    'randomly shuffle dataset'
    ind = np.arange(d.shape[0])
    np.random.shuffle(ind)
    d = d[ind]
    l = l[ind]
    return d, l


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])
    #output = tf.nn.dropout(output, keep_prob=0.80)
    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob=0.50)    # adding dropout after activators
    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob=0.50)     # adding dropout after activators
    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob=0.50)     # adding dropout after activators
    output2 = tf.reshape(output, [-1, 4*4*4*DIM])      # D_
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output2)  #D

    return tf.reshape(output, [-1]), output2

def ctgan_train(args, siteid, d, l, modelfile): 

    'build computation graphs'
    real_data = tf.placeholder(tf.float32, shape=[args.batch, OUTPUT_DIM])
    fake_data = Generator(args.batch)

    disc_real, disc_real_2= Discriminator(real_data)
    disc_real_, disc_real_2_= Discriminator(real_data)
    
    disc_fake,  disc_fake_ = Discriminator(fake_data)
    disc_fake_2,disc_fake_2_ = Discriminator(fake_data)

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    # For saving samples
    fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
    fixed_noise_samples = Generator(128, noise=fixed_noise)

    # number of iterations for training discriminator        
    disc_iters = CRITIC_ITERS
    #original cost
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    wdist = -disc_cost

    #consistency cost
    CT = LAMBDA_2*tf.square(disc_real-disc_real_)
    CT += LAMBDA_2*0.1*tf.reduce_mean(tf.square(disc_real_2-disc_real_2_),reduction_indices=[1])
    CT_ = tf.maximum(CT-Factor_M,0.0*(CT-Factor_M))
    disc_cost += tf.reduce_mean(CT_)

    'GP: gradient penalty'
    alpha = tf.random_uniform(shape=[args.batch,1], minval=0.,maxval=1.)
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates)[0], [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)

    disc_cost += LAMBDA*gradient_penalty

    'training operation for G'
    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    
    'training operation for D'
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    # Train loop    
    saver = tf.train.Saver(max_to_keep=None)
    with tf.Session() as session:

   	'initialize or restore'
	session.run(tf.global_variables_initializer())
        'restore older model'
        if load_dir is not None:
            saver.restore(session, load_dir)
            print 'Model restored'

	iterid = 0
        for i in xrange(args.epochs):
	    'new epoch'
            start_time = time.time()
	    d, l = rshuffle(d, l)
	    wdlist = []  ; # W-Distance
	    dlist = []   ; # Discriminator cost

	    for j in range(0,d.shape[0],args.batch*disc_iters):
	        'update generator (G)'
                if iterid > 0:
                    _ = session.run(gen_train_op)

		iterid += 1 

	        'update discriminator (D)'
                for jj in xrange(disc_iters):
		    _data = d[j+args.batch*jj:j+args.batch*(jj+1)].reshape([args.batch, OUTPUT_DIM])
                    _disc_cost, _wd, _ = session.run([disc_cost, disc_train_op],
		   	                         feed_dict={real_data: _data})
		    wdlist.append(_wd)
		    dlist.append(_disc_cost) 

	    'summarize each epoch'
	    tepoch =  time.time() - start_time
 	    print 'epoch: ', i, iterid, np.mean(dlist), np.mean(wdlist), tepoch

	    'generate sample images'
	    if i % 100 == 99: 
                imgfile = './samples/sample-%03d-of-%03d-%06d.png' % (siteid, args.nsite, i)
                samples = session.run(fixed_noise_samples)
                lib.save_images.save_images(samples.reshape((128, 28, 28)),
				            imgfile)
   	        'save current model'
	        saver.save(session, modelfile, global_step=i)
 	   


# for converting to tf records
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def save_tfrecord(tffile, images, labels, mode):
    '''
        save images to target tfrecords
    '''

    print 'incoming image info: ', images.shape, images.max(), images.min()
    num = images.shape[0]
    if mode == 'gen':
	'[*, 1, 28, 28], ~(0, 1.)' 
#        images = images
        images = images.transpose((0, 2, 3, 1)).reshape((num, -1))
    else: 
	'[*, 28, 28, 1], ~(0, 1.)' 
        images = images.reshape((num, -1))

    print 'start writing to tfrecord: ', tffile
    writer = tf.python_io.TFRecordWriter(tffile)
    for i in range(num):
        image = images[i].tolist()
        image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(28),
            'width': _int64_feature(28),
            'depth': _int64_feature(1),
            'label': _int64_feature(int(labels[i])),
            'image': image_feature}))
        writer.write(example.SerializeToString())
    writer.close()

    print 'completed writing to tfrecord: ', tffile


def ctgan_genimg(args, siteid, modelfile):

    tup = (siteid, args.nsite, args.nimg)
    print 'site %d of %d | generating %d images ...' % tup

    'computational graph'
    gen_noise = tf.placeholder(dtype=tf.float32, shape=[None, 128])
    gen_imgs = Generator(args.genbatch, noise=gen_noise)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, modelfile)
    print 'Model restored'
    
    nrep = args.nimg/args.genbatch
    
    'randomly assign all labels'
    labels = np.random.randint(10, size=(args.nimg, ), dtype=np.int64)  
    images = np.zeros((args.nimg, 1, 28, 28), dtype=np.float32)
    batch = args.genbatch
    for i in xrange(nrep):

        noise = np.random.normal(size=(args.genbatch,128))
        batch_data = sess.run(gen_imgs,feed_dict={gen_noise:noise})    
 	# print batch_data.shape

        images[i*batch:(i+1)*batch] = batch_data.reshape([args.genbatch, 1, 28, 28])

#    print 'save to npy format'
#    fname = 'multi_site/gen-%03d-of-%03d-pct%03d.npy' % (siteid, args.nsite, args.ypct)
#    npfile = os.path.join(args.npdir, fname)
#    np.save(npfile, images)

    print 'save to tfrecords'
    fname = 'multi_site/gen-%03d-of-%03d-pct%03d.tfrecords' % (siteid, args.nsite, args.ypct)
    tffile = os.path.join(args.tfdir, fname)
    save_tfrecord(tffile, images, labels, 'gen')

    print "saving images to %s" % args.imgdir
    if args.saveimg:
        # for  j in xrange(nimg):
        for  j in xrange(100):
            im = np.zeros((28,28,1))
            im[:,:,0]=images[j,0,:,:]*255.
            imgfile = '%s/imgs-%03d-of-%03d-%06d.png' % (args.imgdir, siteid, args.nsite, j)
            cv2.imwrite(imgfile, im)    


def mnist_split(args): 


                                
if __name__ == '__main__':
    # --------- start of main procedures -----------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', help='directory of training dataset',
                                   default='/home/xiaoqzhu/dataset/mnist')
    parser.add_argument('--tfdir', help='target directory of dataset in tfrecord',
                                  default='/home/xiaoqzhu/dataset/mnist')
#    parser.add_argument('--npdir', help='target directory of dataset in *.npy ',
#                                   default='/home/xiaoqzhu/dataset/mnist')
    parser.add_argument('--imgdir', help='target directory of generated images',
                                   default='/home/xiaoqzhu/dataset/mnist/gen/img')

    parser.add_argument('--model', help='generative model location',
                                   default='./models')
    parser.add_argument('--ypct', help='percentage of transimitted label', default=1., type=float)
    parser.add_argument('--batch', help='batch size during training', default=50, type=int)
    parser.add_argument('--epochs', help='training epochs', default=1000, type=int)
    parser.add_argument('--nsite', help='number of sites', default=10, type=int)
    parser.add_argument('--nimg', help='number of generated images', default=60000, type=int)
    parser.add_argument('--genbatch', help='batch size during generation', default=1000, type=int)
    parser.add_argument('--saveimg', default=False, action='store_true')

    args = parser.parse_args()
    print args

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    'load dataset for training'
    tffile = os.path.join(args.datadir, 'train.tfrecords') 
    d, l = load_dataset(args.datadir, 'train')
    d, l = rshuffle(d, l) 
    save_tfrecord(tffile, d, l, 'train')

    nsub=len(d)/args.nsite
    nsample = int(nsub * args.ypct /100.) 
    for i in range(args.nsite): 
	print '==== site %d out of %d ====' % (i, args.nsite)
	dsub = d[i*nsub:(i+1)*nsub]
	lsub = l[i*nsub:(i+1)*nsub]
  	fname = 'multi_site/train-%03d-of-%03d.tfrecords' % (i, args.nsite)
        tffile = os.path.join(args.datadir, fname)
	save_tfrecord(tffile, dsub, lsub, 'train')

  	fname = 'multi_site/sample-%03d-of-%03d-pct%02d.tfrecords' % (i, args.nsite, int(args.ypct))
        tffile = os.path.join(args.datadir, fname)
	save_tfrecord(tffile, dsub[:nsample], lsub[:nsample], 'sample')

	print 'training local GAN'
  	fname = 'model-%03d-of-%03d-ckpt' % (i, args.nsite)
	modelfile = os.path.join(args.model, fname)
        ctgan_train(args, i, dsub, lsub, modelfile)

	print 'generating images from local GAN'
	modelfile = '%s-%d' % (modelfile, args.epochs-1) 
        ctgan_genimg(args, i, modelfile)

    'load and save test dataset'
    tffile = os.path.join(args.datadir, 'test.tfrecords') 
    dt, lt = load_dataset(args.datadir, 'test')
    save_tfrecord(tffile, dt, lt, 'test')

