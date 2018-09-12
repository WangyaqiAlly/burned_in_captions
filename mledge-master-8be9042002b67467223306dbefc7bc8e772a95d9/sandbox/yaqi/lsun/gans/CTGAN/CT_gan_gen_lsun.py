"""CT-WGAN ResNet for CIFAR-10"""
"""highly based on the GP-GAN : https://github.com/igul222/improved_wgan_training """

import os, sys
from sys import stdout
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
#import tflib.cifar10
#import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets
import random
import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')





FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_cls', 20,'how many classes {20,10}')
# tf.app.flags.DEFINE_string('cnn_model_type', 'mid','{small,large,mid}')

tf.app.flags.DEFINE_string('save_dir', "../../Data/generated_data/10percent_site5_model_149999/", "log_dir")
tf.app.flags.DEFINE_string('load_model', "./Record/semigan/10percent_site5/models", "log_dir")



# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = '/home/yaqi/Documents/Data/cifar10/cifar-10-batches-py'    #file path to be modified
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

LAMBDA_2 =2.0  # parameter LAMBDA2
n_examples = 50000 # Number of examples
Factor_M = 0.0  # factor M
BATCH_SIZE = 200 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 100000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 5000 # How frequently to calculate Inception score
MODEL_SAVE_FREQUENCY = 100
CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss
Save_dir = FLAGS.save_dir
load_dir =None

GEN_IMG_NUM=60000





if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=FLAGS.num_cls)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=FLAGS.num_cls)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output

def OptimizedResBlockDisc1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels,kp1,kp2,kp3): # three more parameters of keep rate
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = tf.nn.dropout(output, keep_prob=kp1)     #dropout after activator
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = tf.nn.dropout(output, keep_prob=kp2)     #dropout after activator
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = tf.nn.dropout(output, keep_prob=kp3)     #dropout after activator
    output = nonlinearity(output)
    output2 = tf.reduce_mean(output, axis=[2,3])  #corresponding to D_
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output2)
    output_wgan = tf.reshape(output_wgan, [-1])    #conrresponding to D
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, FLAGS.num_cls, output2)
        return output_wgan, output2, output_acgan
    else:
        return output_wgan, output2, None  # two layers' of output

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_images_and_labels(images, labels, filepath):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)
    for index in range(num_examples):
        image = images[index].tolist()
        image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(32),
            'width': _int64_feature(32),
            'depth': _int64_feature(3),
            'label': _int64_feature(int(labels[index])),
            'image': image_feature}))
        writer.write(example.SerializeToString())
    writer.close()


def save_tfrecords(images,labels,filename):
    assert images.shape[0]==labels.shape[0]
    num=images.shape[0]
    rng = np.random.RandomState(1234)
    rand_ix = rng.permutation(num)
    images = images[rand_ix]
    labels = labels[rand_ix]
    images=images/2.
    images =images.astype(np.float32)
    labels=labels.astype(np.int64)
    images = images.reshape((num, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((num, -1))
    print "save_tfrecord images info:", images.shape,images.max(),images.min()
    convert_images_and_labels(images,  labels, os.path.join(FLAGS.save_dir,filename))






def save_generated_img_semi( batch_num,model,batch_i):
    gen_noise = tf.placeholder(dtype=tf.float32, shape=[None, 128])
    gen_labels = tf.placeholder(dtype=tf.int32, shape=[None])
    gen_imgs = Generator(BATCH_SIZE, gen_labels, noise=gen_noise)
    wdistance, _, disc_label = Discriminator(gen_imgs, gen_labels, 1.0, 1.0, 1.0)
    sess = tf.Session()
    saver = tf.train.Saver()
    if FLAGS.load_model is not None:
        load_model=FLAGS.load_model+'/model.ckpt-'+model
        print 'Model restored from ', load_model #tf.train.latest_checkpoint(FLAGS.load_model)
        saver.restore(sess ,load_model)#tf.train.latest_checkpoint(FLAGS.load_model))
        print 'Model restored'
    images_highwd_top = np.zeros((GEN_IMG_NUM, 3 * 32 * 32), dtype=np.float32)
    labels_highwd_top = np.zeros((GEN_IMG_NUM,), dtype=np.int64)
    for i in xrange(batch_num):
        print "generating batch {}.......".format(i)
        images = np.zeros((GEN_IMG_NUM, 3 * 32 * 32), dtype=np.float32)
        labels = np.zeros((GEN_IMG_NUM,), dtype=np.int64)
        labels_fix = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19] * (GEN_IMG_NUM / 20)
        random.shuffle(labels_fix)
        labels_fix = np.asarray((labels_fix), dtype=np.int64)  # fix the pct of each class
        ac_labels = np.zeros((GEN_IMG_NUM,), dtype=np.int64)
        wdistances = np.zeros((GEN_IMG_NUM,), dtype=np.float32)
        all_noise = np.random.normal(size=(GEN_IMG_NUM, 128))
        for j in range(GEN_IMG_NUM / BATCH_SIZE):  # change 10 to how many batches
            stdout.write("\r%d" % j)
            stdout.flush()
            noise = all_noise[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            batch_labels = labels_fix[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            batch_data, _disc_label, _batch_wdistance = sess.run([gen_imgs, disc_label, wdistance],
                                                                 feed_dict={gen_noise: noise, gen_labels: batch_labels})
            batch_ac_labels = np.argmax(_disc_label, axis=1)
            batch_data = batch_data.astype(np.float32)
            # print batch_data.shape,batch_data.max(),batch_data.min()
            # print batch_labels,batch_labels.max(),batch_labels.min()
            # print batch_ac_labels,batch_ac_labels.max(),batch_ac_labels.min()
            # assert 0
            images[j * BATCH_SIZE:(j + 1) * BATCH_SIZE] = batch_data
            ac_labels[j * BATCH_SIZE:(j + 1) * BATCH_SIZE] = batch_ac_labels
            wdistances[j * BATCH_SIZE:(j + 1) * BATCH_SIZE] = _batch_wdistance

        assert (j + 1) * BATCH_SIZE == GEN_IMG_NUM

        idx_sortbyw = np.argsort(wdistances)
        wdistances = wdistances[idx_sortbyw]
        print wdistances[0], wdistances[-1]
        images_sortbyw = images[idx_sortbyw]
        ac_labels_sortbyw = ac_labels[idx_sortbyw]
        cond_labels_sortbyw = labels_fix[idx_sortbyw]

        agree_idx = np.where(np.equal(ac_labels_sortbyw, cond_labels_sortbyw))
        labels_agree = cond_labels_sortbyw[agree_idx]
        images_agree = images_sortbyw[agree_idx]
        num_agree = len(agree_idx[0])
        print "label agree num:", num_agree
        #
        # save_tfrecords(images_sortbyw[:GEN_IMG_NUM / 2], ac_labels_sortbyw[:GEN_IMG_NUM / 2],
        #                'CTGAN_ac_wdlow_25k_central_batch{}.tfrecords'.format(i))
        # save_tfrecords(images_agree[:num_agree / 2], labels_agree[:num_agree / 2],
        #                'CTGAN_agree_wdlow_{}_central_batch{}.tfrecords'.format(num_agree / 2, i))
        #
        # save_tfrecords(images_sortbyw[GEN_IMG_NUM / 2:], ac_labels_sortbyw[GEN_IMG_NUM / 2:],
        #                'CTGAN_ac_wdhigh_25k_central_batch{}.tfrecords'.format(i))
        # save_tfrecords(images_agree[num_agree / 2:], labels_agree[num_agree / 2:],
        #                'CTGAN_agree_wdhigh_{}_central_batch{}.tfrecords'.format(num_agree / 2, i))

        train_images = np.zeros((6000, 3 * 32 * 32), dtype=np.float32)
        train_labels = np.zeros((6000,), dtype=np.int64)
        examples_per_class = 300
        for k in xrange(20):
            ind = np.where(ac_labels_sortbyw == k)[0]
            print 'label:', k, 'total pos wd img num:', len(ind)
            #print 'label:', k, ind[-examples_per_class:]
            train_images[k * examples_per_class:(k + 1) * examples_per_class] \
                = images_sortbyw[ind[-examples_per_class:]]
            train_labels[k * examples_per_class:(k + 1) * examples_per_class] \
                = ac_labels_sortbyw[ind[-examples_per_class:]]
        assert (k + 1) * examples_per_class == 6000
        #rand_ix_labeled = rng.permutation(5000)
        #train_images, train_labels = train_images[rand_ix_labeled], train_labels[rand_ix_labeled]

        images_highwd_top[i * 6000:(i + 1) * 6000] = train_images
        labels_highwd_top[i * 6000:(i + 1) * 6000] = train_labels
    save_tfrecords(images_highwd_top, labels_highwd_top,
                   'CTGAN_ac_wdtop_60k_batch{}.tfrecords'.format(batch_i))



if __name__ == '__main__':
    models = ['149999']
    batch_num = 10
    save_file_dir = FLAGS.save_dir
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    for model in models:

        # save_generated_img(save_file_dir, batch_num)
        # save_generated_img_uncond(save_file_dir, batch_num,model)
        # save_generated_img_multisites(save_file_dir, batch_num,site='02')
        for batch_i in xrange(10):
            save_generated_img_semi(batch_num,model, batch_i)

