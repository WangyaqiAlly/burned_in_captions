"""CT-WGAN ResNet for CIFAR-10"""
"""highly based on the GP-GAN : https://github.com/igul222/improved_wgan_training """

import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
#import tflib.cifar10
import tflib.inception_score
import tflib.plot
#import sklearn.datasets

import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

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
import read_speechData as speechData
#import create_labeled_unlabeled_data_sets as speechInput

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--m', help='latent space dimensionality', default=256, type=int)
#parser.add_argument('--n', help='number of units per layer', default=32, type=int)
#parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
parser.add_argument('--batch', help='batch size', default=10, type=int)
parser.add_argument('--epochs', help='training epochs', default=200000, type=int)
#parser.add_argument('--model', help='output model', default='model.proto.ali_label2.200k')
#parser.add_argument('--debug', default=False, action='store_true')
#parser.add_argument('--step', help='step to start training', default=0, type=int)
#parser.add_argument('--dataFolder', help='data folder', default='../celebA/img_align_celeba/')
#parser.add_argument('--labelsDir', help='labels directory', default='../celebA/list_attr_celeba.txt')
args = parser.parse_args()
print args


tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]




FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('num_cls', '20','how many classes {20,10}')
# tf.app.flags.DEFINE_string('cnn_model_type', 'mid','{small,large,mid}')

tf.app.flags.DEFINE_string('save_dir', "./Record/GAN_Speech50PercentUnlabeledData/", "log_dir")



# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
#DATA_DIR = args.dataFolder

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

LAMBDA_2 =2.0  # parameter LAMBDA2
Factor_M = 0.0  # factor M
BATCH_SIZE = args.batch # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = args.epochs # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 16384 # Number of pixels in celebA (128*128*1)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 5000 # How frequently to calculate Inception score
MODEL_SAVE_FREQUENCY = 1000
CONDITIONAL = False # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss
Save_dir = FLAGS.save_dir
load_dir =None
NUM_LABELS = 2


if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)
Img_dir = Save_dir+'/samples/'
Figure_dir = Save_dir+'/figures/'
Model_dir = Save_dir+'/models/'
if not os.path.exists(Img_dir):
    os.makedirs(Img_dir)
if not os.path.exists(Figure_dir):
    os.makedirs(Figure_dir)
if not os.path.exists(Model_dir):
    os.makedirs(Model_dir)





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
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=1, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=1, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

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
    output = ResidualBlock('Generator.5', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 1, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels,kp1,kp2,kp3): # three more parameters of keep rate
    output = tf.reshape(inputs, [-1, 1, 128, 128])
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
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output2)
        return output_wgan, output2, output_acgan
    else:
        return output_wgan, output2, None  # two layers' of output





savePath = './Record/GAN_Speech50PercentUnlabeledData/generatedImages/'
if not os.path.exists(savePath):
	os.makedirs(savePath)
modelName = './Record/GAN_Speech50PercentUnlabeledData/models/model.ckpt-63999'

batch_size = 100
numGenData = 18000


gen_noise = tf.placeholder(dtype=tf.float32, shape=[None, 128])
gen_labels = tf.placeholder(dtype=tf.int32, shape=[None])
gen_imgs = Generator(batch_size, gen_labels, noise=gen_noise)
wdistance, _, disc_label = Discriminator(gen_imgs, gen_labels, 1.0, 1.0, 1.0)
sess = tf.Session()
saver = tf.train.Saver()
print 'Model restored from', modelName
saver.restore(sess, modelName)
print 'Model restored'
all_noise = np.random.normal(size=(numGenData, 128))
labels_fix = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * (numGenData / 10)
labels_fix = np.asarray((labels_fix), dtype=np.int64)

for j in range(numGenData//batch_size):
	noise = all_noise[j * batch_size:(j + 1) * batch_size]
	batch_labels = labels_fix[j * batch_size:(j + 1) * batch_size]
	xgen = sess.run([gen_imgs],feed_dict={gen_noise: noise, gen_labels: batch_labels})
	#myNoise = np.random.normal(size=(batch_size, 128))
	#xgen = sess.run(gen_imgs)#, feed_dict={labels:np.zeros(batch_size), noise:myNoise})
	print len(xgen[0]),xgen[0].shape[0]
	for imgNum in range(len(xgen[0])):
		img = xgen[0][imgNum].reshape(1,128,128)
		img = (np.clip(img,0.,1.)*255.).astype('uint8')
		cv2.imwrite(savePath+str(j)+'_'+str(imgNum)+'.png',img.transpose(1,2,0))

#noise = tf.placeholder(dtype=tf.float32, shape=[None, 128])
#fixed_labels = tf.constant(np.array([0,1]*batch_size,dtype='int32'))
#fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
#labels = tf.placeholder(dtype=tf.int32, shape=[None])
#gen_imgs = Generator(batch_size, fake_labels_100)
#saver = tf.train.Saver()

#with tf.Session() as sess:
	#print 'Model restored from', modelName
	#saver.restore(sess, modelName)
	#print 'Model restored'
	#for i in range(300):
		#myNoise = np.random.normal(size=(batch_size, 128))
	#	xgen = sess.run(gen_imgs)#, feed_dict={labels:np.zeros(batch_size), noise:myNoise})
#		for img in xgen:
#			img = img.reshape(64,64,3)
#			cv2.imwrite(savePath+str(i)+'.png',img)#.transpose(1,2,0))





