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
import tflib.ops.deconv2d
import tflib.save_images

'Fixed parameters'
Factor_M = 0.0    # factor M
LAMBDA   = 10. # Gradient penalty lambda hyperparameter
LAMBDA_2 = 2.0    # weight factor
ACGAN_SCALE = 1.
ACGAN_SCALE_G = 0.1
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


def Batchnorm(name, axes, inputs, 
	      # is_training=None, 
              # stats_iter=None, 
	      # update_moving_stats=True, 
	      # fused=True, 
	      labels=None, 
	      n_labels=None):

    """
	conditional batchnorm (dumoulin et al 2016)
	for BCHW conv filtermaps
    """

    print 'inputs: ', inputs
    print 'labels: ', labels
    if axes != [0,2,3]:
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    print 'mean:', mean
    print 'var:', var

    shape = mean.get_shape().as_list() # shape is [1,n,1,1]
    print 'shape: ', shape

    offset_m = lib.param(name+'.offset', np.zeros([n_labels,shape[1]], dtype='float32'))
    scale_m = lib.param(name+'.scale', np.ones([n_labels,shape[1]], dtype='float32'))
    print 'offset_m: ', offset_m
    print 'scale_m: ', scale_m

    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    print 'offset: ', offset
    print 'scale: ', scale

    result = tf.nn.batch_normalization(inputs, mean, var, offset[:,:,None,None], scale[:,:,None,None], 1e-5)
    return result

def Generator(n_samples, noise=None, labels=None):

    # incoming noise dimension: 128
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    print 'shape of noise: ', noise.shape
    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])
    print 'G | ouptput', output

    # deconv 5x5
    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    print 'G.2 | ouptput', output
    if args.semi:
        # output = lib.ops.cond_batchnorm.Batchnorm('Generator',[0,2,3],output,labels=labels,n_labels=10)
        output = Batchnorm('Generator',[0,2,3],output,labels=labels,n_labels=10)
    output = tf.nn.relu(output)
    print 'G | ouptput', output

    # clip to 7x7
    output = output[:,:,:7,:7]
    print 'G | ouptput', output

    # decov 5x5
    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    print 'G.3 | ouptput', output
    # if args.semi:
        # output = lib.ops.cond_batchnorm.Batchnorm('Generator',[0,2,3],output,labels=labels,n_labels=10)
        # output = Batchnorm('Generator',[0,2,3],output,labels=labels,n_labels=10)
    output = tf.nn.relu(output)
    print 'G | ouptput', output

    # deconv 5x5
    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)
    print 'G.5 | ouptput', output

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
    output_wgan = tf.reshape(output, [-1])    #conrresponding to D

    if args.semi: 
	output_acgan = lib.ops.linear.Linear('Discriminator.ACOutput', 4*4*4*DIM, 10, output2)
        return output_wgan, output2, output_acgan
    else: 	
        return output_wgan, output2, None

def read(filename_queue,train):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([784], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    image = features['image']
    # image = tf.reshape(image, [28, 28, 1])
    # label = tf.one_hot(tf.cast(features['label'], tf.int32), 10)
    label = tf.cast(features['label'], tf.int32)
    return image, label

def load_real_data(tffile, batch_size=100, nsample=6000, num_epochs=None):

    print 'load testing dataset'
    filenames = [tffile]
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
    image, label = read(filename_queue, True)

    print 'loading real data:', filenames
    print 'image shape: ', image.shape
    print 'lable shape: ', label.shape
    print 'max pixel value: ', np.amax(image)
    print 'min pixel value: ', np.amin(image)
    
    return tf.train.shuffle_batch([image, label],
                                  batch_size=batch_size,
                                  num_threads=4,
                                  capacity=nsample + 3 * batch_size,
                                  min_after_dequeue=nsample)

def ctgan_train(args, siteid, modelfile): 

    print 'training CT-GAN'
    print 'target model file: ', modelfile

    'build computation graphs'
    fname = 'multi_site/train-%03d-of-%03d.tfrecords' % (siteid, args.nsite)
    tffile = os.path.join(args.datadir, fname)
    nsample = 60000/args.nsite
    # real_data = tf.placeholder(tf.float32, shape=[args.batch, OUTPUT_DIM])
    real_data, real_label = load_real_data(tffile, batch_size=args.batch, nsample=nsample)
    fake_label = real_label 
    fake_data = Generator(args.batch, labels=fake_label)


    print 'real_label: ', real_label
    print 'real_data:  ', real_data
    print 'fake_label: ', real_label
    print 'fake_data:  ', fake_data
    print '............................'
    disc_real, disc_real_2, disc_real_ac,  = Discriminator(real_data)
    disc_real_, disc_real_2_, disc_real_ac_ = Discriminator(real_data) 
    print 'disc_real:',      disc_real
    print 'disc_real_: ',    disc_real_
    print 'disc_real_2:',    disc_real_2
    print 'disc_real_2_: ',  disc_real_2_
    print 'disc_real_ac:',   disc_real_ac
    print 'disc_real_ac_: ', disc_real_ac_
    print '............................'
    disc_fake,  disc_fake_2, disc_fake_ac = Discriminator(fake_data)
    # disc_fake_2,disc_fake_2_ = Discriminator(fake_data)
    print 'disc_fake: ', disc_fake
    print 'disc_fake_2: ',  disc_fake_2
    print 'disc_fake_ac:',   disc_fake_ac

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    # For saving samples
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator(100, noise=fixed_noise, labels=fixed_labels)

    print 'fixed_noise: ', fixed_noise_samples
    
    #generator cost
    gen_cost = -tf.reduce_mean(disc_fake)

    # discriminator cost
    wdist = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
    disc_cost = -wdist

    'GP: gradient penalty'
    alpha = tf.random_uniform(shape=[args.batch,1], minval=0.,maxval=1.)
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates)[0], [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    'CT: consistency cost'
    CT =      tf.square(disc_real-disc_real_)
    CT += 0.1*tf.reduce_mean(tf.square(disc_real_2-disc_real_2_),reduction_indices=[1])
    CT_ = tf.maximum(CT-Factor_M,  0.0)
    disc_cost += LAMBDA_2 *tf.reduce_mean(CT_)

    print '... cost functions ...'
    print 'wdist: ', wdist
    print 'gcost: ', gen_cost
    print 'dcost: ', disc_cost
    print '..................... '

    'AC output for semi-supervised learning'
    if args.semi:
	logent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_real_ac, 
							        labels=real_label) 
	disc_cost += ACGAN_SCALE*tf.reduce_mean(logent)

	gen_ac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_ac, 
							        labels=fake_label) 
	gen_cost += ACGAN_SCALE_G*tf.reduce_mean(gen_ac)

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

        'Create a coordinator and run all QueueRunner objects'
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

	wdistlist = []  ; # W-Distance
	dcostlist = []   ; # Discriminator cost
        gcostlist = []   ; # generator cost
	for i in xrange(args.niter):
	    print 'generator iter %d ...' % i
            start_time = time.time()
	    # d, l = rshuffle(d, l)

	    print 'update generator (G)'
            if i > 0:
	    	_gcost, _ = session.run([gen_cost, gen_train_op])
	    else: 
		_gcost = 0.

	    print 'update discriminator (D)'
            for j in xrange(CRITIC_ITERS):
		print '#'*(j+1),	
                _dcost, _wdist, _ = session.run([disc_cost, wdist, disc_train_op])
		print 'dcost = ', _dcost, 'wdist = ', _wdist

	    gcostlist.append(_gcost)
            wdistlist.append(_wdist)
	    dcostlist.append(_dcost)
	    print 'generator iter: %d' % i, _gcost, _dcost, _wdist

	    if i % 1000 == 999: 
	        'generate sample images'
                imgfile = './samples/sample-%03d-of-%03d-%06d.png' % (siteid, args.nsite, i)
                samples = session.run(fixed_noise_samples)
                lib.save_images.save_images(samples.reshape((100, 28, 28)),
				            imgfile)
   	        'save current model'
	        saver.save(session, modelfile, global_step=i)

	print '... end of all generator iterations ...'
   	if args.semi: 
	    txtfile = './ctgan-semi-%03d-of-%03d-sites.txt' % (siteid, args.nsite)
 	else: 
	    txtfile = './ctgan-uncond-%03d-of-%03d-sites.txt' % (siteid, args.nsite)
	mat = np.zeros((args.niter, 3))
	mat[:,0]=np.asarray(gcostlist).reshape((args.niter))
	mat[:,1]=np.asarray(dcostlist).reshape((args.niter))
	mat[:,2]=np.asarray(wdistlist).reshape((args.niter))
	np.savetxt(txtfile, mat, fmt='%6.2f')	

	 # Stop the threads
        coord.request_stop()
    
        # Wait for threads to stop
        coord.join(threads)
    

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
    print 'target modelfile: ', modelfile

    batch = args.genbatch
    'computational graph'
    gen_noise = tf.placeholder(dtype=tf.float32, shape=[None, 128])
    gen_labels = tf.placeholder(dtype=tf.int32, shape=[None])
    gen_imgs = Generator(batch, noise=gen_noise, labels=gen_labels)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, modelfile)
    print 'Model restored'
    
    nrep = args.nimg/batch
    
    'randomly assign all labels'
    labels = np.random.randint(10, size=(args.nimg, ), dtype=np.int64)  
    labels = np.zeros((args.nimg, ), dtype=np.int64)  
    images = np.zeros((args.nimg, 1, 28, 28), dtype=np.float32)
    for i in xrange(nrep):
        labels_in=np.asarray([0,1,2,3,4,5,6,7,8,9]*(batch/10), dtype=np.int64) 
        noise = np.random.normal(size=(batch,128))
        batch_data = sess.run(gen_imgs,feed_dict={gen_noise:noise, gen_labels:labels_in})  
 	# print batch_data.shape

        images[i*batch:(i+1)*batch] = batch_data.reshape([batch, 1, 28, 28])
        labels[i*batch:(i+1)*batch] = labels_in

#    print 'save to npy format'
#    fname = 'multi_site/gen-%03d-of-%03d-pct%03d.npy' % (siteid, args.nsite, args.ypct)
#    npfile = os.path.join(args.npdir, fname)
#    np.save(npfile, images)

    print 'save to tfrecords'
    if args.semi: 
        fname = 'multi_site/gen-%03d-of-%03d-semi.tfrecords' % (siteid, args.nsite)
    else: 
        fname = 'multi_site/gen-%03d-of-%03d.tfrecords' % (siteid, args.nsite)
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


def mnist_split(args, d, l, ypctlist): 
    '''
    split dataset into N sites, and save
    fraction of samples per site'
    '''

    'shuffle original dataset first'    
    d, l = rshuffle(d, l) 
    nsub=len(d)/args.nsite   # number of samples per site
 
    for i in range(args.nsite): 
	print '==== site %d out of %d ====' % (i, args.nsite)
	dsub = d[i*nsub:(i+1)*nsub]
	lsub = l[i*nsub:(i+1)*nsub]
  	fname = 'multi_site/train-%03d-of-%03d.tfrecords' % (i, args.nsite)
        tffile = os.path.join(args.datadir, fname)
	save_tfrecord(tffile, dsub, lsub, 'train')

	for j, ypct in enumerate(ypctlist): 
            nsample = int(nsub * ypct /100.) 
     	    fname = 'multi_site/sample-%03d-of-%03d-pct%02d.tfrecords' % (i, args.nsite, int(ypct))
            tffile = os.path.join(args.datadir, fname)
	    save_tfrecord(tffile, dsub[:nsample], lsub[:nsample], 'sample')
                                
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
    parser.add_argument('--semi', help='semi-supervised learning', default=False, action='store_true')
    parser.add_argument('--batch', help='batch size during training', default=60, type=int)
    parser.add_argument('--niter', help='number of generator training iterations', 
				   default=50000, type=int)
    parser.add_argument('--nsite', help='number of sites', default=10, type=int)
    parser.add_argument('--nimg', help='number of generated images', default=60000, type=int)
    parser.add_argument('--genbatch', help='batch size during generation', default=1000, type=int)
    parser.add_argument('--saveimg', help='save generated images', default=False, action='store_true')

    args = parser.parse_args()
    print args

#    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#    'load and save test dataset'
#    tffile = os.path.join(args.datadir, 'test.tfrecords') 
#    dt, lt = load_dataset(args.datadir, 'test')
#    save_tfrecord(tffile, dt, lt, 'test')

#    'load dataset for training'
#     tffile = os.path.join(args.datadir, 'train.tfrecords') 
#    d, l = load_dataset(args.datadir, 'train')
#    save_tfrecord(tffile, d, l, 'train')

#    'split data into multiple sites'
#    ypctlist = [1., 2., 5., 10., 20.]
#    mnist_split(args, d, l, ypctlist)

    'train and generate image per site'
    for i in range(args.nsite): 
    # for i in range(1, args.nsite): 
	print '==== site %d out of %d ====' % (i, args.nsite)
	print 'training local GAN'
  	fname = 'model-%03d-of-%03d-ckpt' % (i, args.nsite)
	modelfile = os.path.join(args.model, fname)
        ctgan_train(args, i, modelfile)

	print 'generating images from local GAN'
	modelfile = '%s-%d' % (modelfile, args.niter-1) 
        ctgan_genimg(args, i, modelfile)

