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
locale.setlocale(locale.LC_ALL, '')

label_prior0 = np.array([0.5,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
label_prior0 = label_prior0/label_prior0.sum()

label_prior1 = 1.0-np.array([0.5,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
label_prior1 = label_prior1/label_prior1.sum()


Data_rate=1.0
######## CHANGE THIS TO CHANGE THE SITE NUMBER!!!
USE_GENERATOR = True
# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = './unbanlance_data/'
TEST_DATA_DIR = '../improved_wgan_training/Data/cifar-10-batches-py/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 1000000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 1000 # How frequently to calculate Inception score
MODEL_SAVE_FREQUENCY = 10000 # How frequently to calculate Inception score
load_dir_0 = './Record/site_0/models/model.ckpt-69999'
load_dir_1 = './Record/site_1/models/model.ckpt-69999'
Save_dir = './Record/'
Img_dir = Save_dir+'samples/'
Model_dir = Save_dir+'models/'
if not os.path.exists(Img_dir):
    os.makedirs(Img_dir)
if not os.path.exists(Model_dir):
    os.makedirs(Model_dir)

CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss

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

def Generator(n_samples, labels,Site_N, noise=None):
    # if noise is not None:
    #     noise = noise
    # else:
    #     noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear('site_'+str(Site_N)+'/'+'Generator.Input', 128, 4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('site_'+str(Site_N)+'/'+'Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('site_'+str(Site_N)+'/'+'Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('site_'+str(Site_N)+'/'+'Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('site_'+str(Site_N)+'/'+'Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('site_'+str(Site_N)+'/'+'Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None

def OptimizedResBlockDisc1_classifier(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Classifier.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Classifier.1.Conv1', filter_size=3, inputs=output)    
    output = nonlinearity(output)            
    output = conv_2('Classifier.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def Classifier(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1_classifier(output)
    output = ResidualBlock('Classifier.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Classifier.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Classifier.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Classifier.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Classifier.ACGANOutput', DIM_D, 10, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None




_iteration = tf.placeholder(tf.int32, shape=None)
x = tf.placeholder(tf.float32, shape=[2*BATCH_SIZE, OUTPUT_DIM])
l = tf.placeholder(tf.int32, shape=[2*BATCH_SIZE])
fake_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

# gen_labels = tf.cast(tf.random_uniform([BATCH_SIZE])*10, tf.int32)
noise = tf.random_normal([BATCH_SIZE, 128])
gen_imgs_0 = Generator(BATCH_SIZE, fake_labels, noise=noise,Site_N=0)
gen_imgs_1 = Generator(BATCH_SIZE, fake_labels, noise=noise,Site_N=1)
_,disc_label = Discriminator(x,labels=0)
acc_disc = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.to_int32(tf.argmax(disc_label, dimension=1)),l
                    ),
                    tf.float32
                )
            )


 

disc_w, disc_ac = Classifier(x, l)
closs = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_ac, labels=l))
acc = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.to_int32(tf.argmax(disc_ac, dimension=1)),l
                    ),
                    tf.float32
                )
            )


class_params = lib.params_with_name('Classifier.')
gen_params_0 = lib.params_with_name('site_0/Generator.')
gen_params_1 = lib.params_with_name('site_1/Generator.')
disc_params = lib.params_with_name('Discriminator.')

if DECAY:
    decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
else:
    decay = 1.
class_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
class_gv = class_opt.compute_gradients(closs, var_list=class_params)
class_train_op = class_opt.apply_gradients(class_gv)


sess = tf.Session()

    # samples = ((samples+1.)*(255./2)).astype('int32')
    # lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), Img_dir+'samples_{}.png'.format(frame))

train_gen, dev_gen = lib.cifar10.load(2*BATCH_SIZE, DATA_DIR,0,testing=True,test_dir = TEST_DATA_DIR)
def inf_train_gen():
    while True:
        for images,_labels in train_gen():
            yield images,_labels


saver_class = tf.train.Saver(var_list=class_params)
saver_gen_0 = tf.train.Saver(var_list=gen_params_0)
saver_gen_1 = tf.train.Saver(var_list=gen_params_1)
# saver_disc = tf.train.Saver(var_list=disc_params)
sess.run(tf.initialize_all_variables())

saver_gen_0.restore(sess,load_dir_0)
saver_gen_1.restore(sess,load_dir_1)
# saver_disc.restore(sess,load_dir_0)
print 'Model restored'

dec_acc=[]
# for images,_labels in dev_gen():
#     images_float = np.reshape(2*((images.astype(np.float32)/256.)-.5), (-1, OUTPUT_DIM))
#     images_float += np.random.uniform(size=(BATCH_SIZE,OUTPUT_DIM),low=0.,high=1./128)
#     _dic_acc = sess.run([acc_disc], feed_dict={x: images_float,l:_labels})
#     dec_acc.append(_dic_acc)
disc_acc = np.mean(dec_acc)

def inf_train_gen():
    while True:
        for images,_labels in train_gen():
            yield images,_labels
gen = inf_train_gen()

for iteration in xrange(ITERS):
    start_time = time.time()

    if USE_GENERATOR:
        fake_labels_0=np.random.choice(np.asarray(xrange(10)),BATCH_SIZE,p=label_prior0)
        gen_img_0 = sess.run(gen_imgs_0,feed_dict={fake_labels:fake_labels_0})
        fake_labels_1=np.random.choice(np.asarray(xrange(10)),BATCH_SIZE,p=label_prior1)
        gen_img_1 = sess.run(gen_imgs_1,feed_dict={fake_labels:fake_labels_1})
        gen_img_ = np.concatenate((gen_img_0,gen_img_1),axis=0)
        fake_labels_ = np.concatenate((fake_labels_0,fake_labels_1),axis=0)
        rng_state = np.random.get_state()
        np.random.shuffle(gen_img_)
        np.random.set_state(rng_state)
        np.random.shuffle(fake_labels_)


        _class_cost, _class_acgan_acc, _ = sess.run([closs, acc, class_train_op], feed_dict={_iteration:iteration,x:gen_img_,l:fake_labels_})
    else:
        _data_images,_data_labels = gen.next()
        _data_float = np.reshape(2*((_data_images.astype(np.float32)/256.)-.5), (-1, OUTPUT_DIM))
        _data_float += np.random.uniform(size=(BATCH_SIZE,OUTPUT_DIM),low=0.,high=1./128)
        _class_cost, _class_acgan_acc, _ = sess.run([closs, acc, class_train_op], feed_dict={_iteration:iteration,x:_data_float,l:_data_labels})

    if iteration % 100 == 99:
        dev_disc_costs = []
        dec_acc=[]
        for images,_labels in dev_gen():
            images_float = np.reshape(2*((images.astype(np.float32)/256.)-.5), (-1, OUTPUT_DIM))
            images_float += np.random.uniform(size=(2*BATCH_SIZE,OUTPUT_DIM),low=0.,high=1./128)

            _clss_disc_cost,_class_acc = sess.run([closs,acc], feed_dict={x: images_float,l:_labels})
            dev_disc_costs.append(_clss_disc_cost)
            dec_acc.append(_class_acc)
        print  '\n','data_rate:',Data_rate,'USE_GENERATOR:',USE_GENERATOR,'iter:',iteration,'  classifier train loss:', _class_cost, '  classifier train acc:', _class_acgan_acc, '  test loss:', np.mean(dev_disc_costs), '  test acc:', np.mean(dec_acc), '  disc test acc:', disc_acc

    if iteration % 1000 == 999:
        if USE_GENERATOR:
            lib.plot.plot('dev_accuracy_use_generator', np.mean(dec_acc))
        else:
            lib.plot.plot('dev_accuracy_use_dataset', np.mean(dec_acc))

    if iteration % MODEL_SAVE_FREQUENCY == MODEL_SAVE_FREQUENCY-1:
        saver_class.save(sess, Model_dir+'model.ckpt', global_step=iteration)

    if (iteration < 500) or (iteration % 1000 == 999):
        lib.plot.flush()

    lib.plot.tick()