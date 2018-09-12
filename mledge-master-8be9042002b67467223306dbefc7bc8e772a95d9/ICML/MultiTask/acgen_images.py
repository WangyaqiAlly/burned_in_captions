"""CT-WGAN ResNet for CIFAR-10"""
"""highly based on the GP-GAN : https://github.com/igul222/improved_wgan_training """

import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm

import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

from read_celebA_3 import derive_input_dir_name

import argparse
import struct
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
parser.add_argument('--n', help='number to generate', default=180000, type=int)
parser.add_argument('--label', help='conditional generation', default='None')
#parser.add_argument('--tag', help='experiment tag', default='orig')
parser.add_argument('--nm', help='number of models to use', default=1, type=int)
parser.add_argument('--percent', help='percent of labelled data', default=10, type=int)
parser.add_argument('--sites', help='num of site', default=-1, type=int)
args = parser.parse_args()
print args


output_path_prefix = '/home2/dataset/Celeb-A'
notused_1, generated_dir, notused_2 = derive_input_dir_name('', args.label)

CONDITIONAL = True

output_path = os.path.join(output_path_prefix, generated_dir)
if args.label == 'Smiling':
    tag = 'Smiling'
    #model_path = 'Record/Smiling/models'
elif args.label == 'Male':
    tag = 'Male'
    #model_path = 'Record/Male/models'
elif args.label == 'None':
    tag = 'Unconditional'
    #model_path = 'Record/Unconditional/models'
    CONDITIONAL = False
elif args.label == 'High_Cheekbones':
    tag = 'High_Cheekbones'
    #model_path = 'Record/High_Cheekbones/models'
else:
    print 'FATAL error: --tag for orig or front must be provided'
    sys.exit(0)

if args.percent <> 10:
    tag = tag + '-' + str(args.percent)

model_path = 'Record/' + tag + '/models'

if not os.path.isdir(output_path):
    os.makedirs(output_path)


DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072*4 # Number of pixels in celebA (64*64*3)
#CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning

NUM_LABELS = 2

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

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
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=NUM_LABELS)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=NUM_LABELS)
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
    output = ResidualBlock('Generator.4', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels,kp1,kp2,kp3): # three more parameters of keep rate
    output = tf.reshape(inputs, [-1, 3, 64, 64])
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
        # !!!!!!!! dtan: this should be a 2, but we made a mistake in GAN, so this has to follow...
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 2, output2)
        #output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output2)
        return output_wgan, output2, output_acgan
    else:
        return output_wgan, output2, None  # two layers' of output


batch_size = 100

gen_noise = tf.placeholder(dtype=tf.float32, shape=[None, 128])
gen_labels = tf.placeholder(dtype=tf.int32, shape=[None])
gen_imgs = Generator(batch_size, gen_labels, noise=gen_noise)
wdistance, _, disc_label = Discriminator(gen_imgs, gen_labels, 1.0, 1.0, 1.0)

def gen_from_model(chkpt, the_sess, the_saver, npoints, phase, skip):
    cur_model_path = os.path.join(model_path, 'model.ckpt-' + str(chkpt))
    print 'Model restored from:', cur_model_path
    the_saver.restore(the_sess, cur_model_path)

    all_noise = np.random.normal(size=(npoints, 128))
    labels_fix = [0,1] * (npoints / 2)
    labels_fix = np.asarray((labels_fix), dtype=np.int64)

    for j in range(npoints//batch_size):
        start = j*batch_size
        noise = all_noise[start:(start+ batch_size)]
        batch_labels = labels_fix[start:(start+batch_size)]
        if CONDITIONAL:
            xgen, glabels = sess.run([gen_imgs, disc_label],feed_dict={gen_noise: noise, gen_labels: batch_labels})
            for imgNum in range(len(xgen)):
                img = xgen[imgNum].reshape(3,64,64)
                #img = (np.clip(img,0.,1.)*255.).astype('uint8')
                img = ((img+1.)*(255./2)).astype('uint8')
                img_tag = '{:06d}'.format(phase + (start + imgNum)*skip)
                #if glabels[imgNum][0] > glabels[imgNum][1]:
                #    cv2.imwrite(output_path_y + '/' +img_tag+'.png',img.transpose(1,2,0))
                #else:
                #    cv2.imwrite(output_path_o + '/' +img_tag+'.png',img.transpose(1,2,0))
                cv2.imwrite(os.path.join(output_path, img_tag + '.png'), img.transpose(1,2,0))
        else:
            xgen = sess.run([gen_imgs], feed_dict={gen_noise: noise, gen_labels: batch_labels})
            all_images = xgen[0].reshape(batch_size, 3,64,64)
            for imgNum in range(batch_size):
                img = all_images[imgNum]
                img = ((img+1.)*(255./2)).astype('uint8')
                img_tag = '{:06d}'.format(phase + (start + imgNum)*skip)
                cv2.imwrite(os.path.join(output_path, img_tag + '.png'), img.transpose(1,2,0))

def gen_from_site(model_dir, the_sess, the_saver, npoints, phase, skip):
    #cur_model_path = os.path.join(model_dir, 'model.ckpt-' + str(chkpt))
    cur_model_path = tf.train.latest_checkpoint(model_dir)
    print 'Model restored from:', cur_model_path
    the_saver.restore(the_sess, cur_model_path)

    all_noise = np.random.normal(size=(npoints, 128))
    labels_fix = [0,1] * (npoints / 2)
    labels_fix = np.asarray((labels_fix), dtype=np.int64)

    for j in range(npoints//batch_size):
        start = j*batch_size
        noise = all_noise[start:(start+ batch_size)]
        batch_labels = labels_fix[start:(start+batch_size)]
        if CONDITIONAL:
            xgen, glabels = sess.run([gen_imgs, disc_label],feed_dict={gen_noise: noise, gen_labels: batch_labels})
            for imgNum in range(len(xgen)):
                img = xgen[imgNum].reshape(3,64,64)
                #img = (np.clip(img,0.,1.)*255.).astype('uint8')
                img = ((img+1.)*(255./2)).astype('uint8')
                img_tag = '{:06d}'.format(phase + (start + imgNum)*skip)
                #if glabels[imgNum][0] > glabels[imgNum][1]:
                #    cv2.imwrite(output_path_y + '/' +img_tag+'.png',img.transpose(1,2,0))
                #else:
                #    cv2.imwrite(output_path_o + '/' +img_tag+'.png',img.transpose(1,2,0))
                cv2.imwrite(os.path.join(output_path, img_tag + '.png'), img.transpose(1,2,0))
        else:
            xgen = sess.run([gen_imgs], feed_dict={gen_noise: noise, gen_labels: batch_labels})
            all_images = xgen[0].reshape(batch_size, 3,64,64)
            for imgNum in range(batch_size):
                img = all_images[imgNum]
                img = ((img+1.)*(255./2)).astype('uint8')
                img_tag = '{:06d}'.format(phase + (start + imgNum)*skip)
                cv2.imwrite(os.path.join(output_path, img_tag + '.png'), img.transpose(1,2,0))

sess = tf.Session()
saver = tf.train.Saver()

if args.sites >= 0:
    for i in range(args.sites):
        gen_from_site('Record/'+tag+'-site'+str(i) + '/models', sess, saver, args.n/args.sites, i, args.sites)
else:
    latest_path = os.path.basename(tf.train.latest_checkpoint(model_path)).split('-')
    chkpt = int(latest_path[1])
    for i in range(args.nm):
        gen_from_model(chkpt-i*5000, sess, saver, (args.n+args.nm-1)/args.nm, i, args.nm)
