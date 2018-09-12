# CUDA_VISIBLE_DEVICES='1' python cifar10_wacage_filter.py
"""WGAN-GP ResNet for CIFAR-10"""

import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets

import time
import functools
import locale
import logging
from resnet import *
# from datetime import datetime
from cifar10_input import *

locale.setlocale(locale.LC_ALL, '')
logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

Site_N = -1
######## CHANGE THIS TO CHANGE THE SITE NUMBER!!!

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = './data/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

Data_rate = 1.0
BATCH_SIZE = 128 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 70000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 1000 # How frequently to calculate Inception score
MODEL_SAVE_FREQUENCY = 5000 # How frequently to calculate Inception score
PLOT_FREQUENCY = 500
FILTER_BAR=0.9
load_dir = '/home/guest/ryang2/AGE/cifar10/wage_resnet/data_generation/match21_1.0/model.ckpt-69999'
Save_dir = './match21_gfilter_'+str(FILTER_BAR)+'/data/'
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)


CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss


def visualization_color(images, labels, edge_id='', fake_label=None, save_img=False, img_dir='', imgs=None):
    samples_per_class = 3
    generated_label = labels
    classes = np.unique(generated_label)
    # show_label= (-1) * np.ones((samples_per_class, len(classes)))
    show_label = None
    # index = 0
    # For each class, choose `samples_per_class` images to show
    for _class in classes:
        one_class_all_ids = np.where(generated_label == _class)[0]
        if len(one_class_all_ids) != 0:
            one_class_num_ids = np.random.choice(one_class_all_ids, samples_per_class)  # .tolist()
            # print type(one_class_num_ids[0])
            if imgs is None:
                imgs = np.concatenate(images[one_class_num_ids], axis=0)
                if fake_label is not None:
                    show_label = [fake_label[one_class_num_ids]]
            else:
                imgs = np.concatenate((imgs, np.concatenate(images[one_class_num_ids], axis=0)), axis=1)
                if fake_label is not None:
                    show_label = np.concatenate([show_label, [fake_label[one_class_num_ids]]], axis=0)
                   
    # print "show label:", show_label.shape, "class num: ", len(classes)
    # Generate the image, append the label, and show
    img = imgs.astype('uint8')
    width = img.shape[0]
    length = img.shape[1]
    img = cv2.resize(img, (length * 3, width * 3), interpolation=cv2.INTER_CUBIC)
    for i, k in enumerate(classes):
        cv2.putText(img, "{:d}".format(k), (i * 32 * 3, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
   
    if fake_label is not None:
        for i, _class in enumerate(classes):
            for _image in xrange(samples_per_class):
                cv2.putText(img, "{:d}".format(show_label[i, _image]), (i * 32 * 3, _image * 32 * 3 + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('Images' + edge_id, img)

    if save_img:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = img_dir + '/' + edge_id + '_{:%Y.%m.%d.%H.%M.%S.%f}.png'.format(datetime.datetime.now())
        cv2.imwrite(img_path, img)
    cv2.waitKey(10)
    return img

def match_l2(x, y):
    return tf.sqrt(tf.reduce_mean(tf.pow(x-y, 2)))

def match_l1(x, y):
    return tf.reduce_mean(tf.abs(x-y))

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
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
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
    return tf.reshape(output, [-1, OUTPUT_DIM]), noise

def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    z = lib.ops.linear.Linear('Discriminator.z', DIM_D, DIM_D, output)
    output = nonlinearity(z)
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)

        return output_wgan, output_acgan, z
    else:

        return output_wgan, None, z
   

with tf.Session() as sess:

    all_gen_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    generate_samples, generate_noises = Generator(BATCH_SIZE, all_gen_labels)
    # _, _, _ = Discriminator(generate_samples, )

    gen_params = lib.params_with_name('Generator.')
    saver = tf.train.Saver(var_list=gen_params)
    saver.restore(sess, load_dir)

    test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 32, 32, 3])
    logits = inference(test_image_placeholder, 5, reusef=False, id='Site_100_')
    predictions = tf.nn.softmax(logits)

    subset_classifier = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Site_100_*')
    subset_classifier_saver = tf.train.Saver(var_list=subset_classifier)
    subset_classifier_saver.restore(sess, '100_percent_golden_models/model.ckpt-79999')
    print 'model restored from 100_percent_golden_models/model.ckpt-79999'

    counter = 0
    

    for i in xrange(600):
        sample_filter = []
        images_filter=[]
        labels_filter = []
        for j in xrange(1000):
            batch_num = i * 1000 + j
            batch_data = np.load('/home/guest/ryang2/AGE/cifar10/wage_resnet/data_generation/match21_1.0/data/img'+('%06d' % batch_num)+'.npy')
            batch_label = np.load('/home/guest/ryang2/AGE/cifar10/wage_resnet/data_generation/match21_1.0/data/logits'+('%06d' % batch_num)+'.npy')
            batch_data1 = batch_data.astype(np.float).reshape((-1,3,32,32)).transpose(0,2,3,1)
            batch_data1 = batch_data1[...,[2,1,0]]
            # print 'batch_data1: ', batch_data1.shape
            l = sess.run(predictions,feed_dict={test_image_placeholder: batch_data1})

            for k in xrange(BATCH_SIZE):
                if l[k, batch_label[k]] > 0.9:
                    sample_filter.append(batch_data[k])
                    images_filter.append(batch_data1[k])
                    labels_filter.append(batch_label[k])

        print i * 1000, len(labels_filter)

            # visualization_color(np.array(images_filter), np.array(labels_filter))
            # time.sleep(3)

        for s in range(0, len(labels_filter)-127, 128):
            np.save(Save_dir+'img'+('%06d' % counter), sample_filter[s:s+128])
            np.save(Save_dir+'logits'+('%06d' % counter), labels_filter[s:s+128])
            counter += 1



