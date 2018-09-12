# CUDA_VISIBLE_DEVICES='0' python wage_resnet_multidisplay.py
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
import cv2

import numpy as np
import tensorflow as tf
import sklearn.datasets

import time
import functools
import locale
import logging
locale.setlocale(locale.LC_ALL, '')
logging.basicConfig(level=logging.DEBUG)

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
BATCH_SIZE = 64 # Critic batch size
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
PLOT_FREQUENCY = 200
load_dir = None
Save_dir = './wage_resnet_multidisplay/rate_'+str(Data_rate)+'/'
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

# def Discriminator_orig(inputs, labels):
#     output = tf.reshape(inputs, [-1, 3, 32, 32])
#     output = OptimizedResBlockDisc1(output)
#     output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
#     output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
#     output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
#     output = nonlinearity(output)
#     output = tf.reduce_mean(output, axis=[2,3])
#     output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
#     output_wgan = tf.reshape(output_wgan, [-1])
#     if CONDITIONAL and ACGAN:
#         output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)
#         return output_wgan, output_acgan, output
#     else:
#         return output_wgan, None, output


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


with tf.Session() as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    fake_data_splits = []
    noise_splits = []
    for i, device in enumerate(DEVICES):
        with tf.device(device):
            fake_data, noise = Generator(BATCH_SIZE/len(DEVICES), labels_splits[i])
            fake_data_splits.append(fake_data)
            noise_splits.append(noise)


    all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

    DEVICES_B = DEVICES[:len(DEVICES)/2]
    DEVICES_A = DEVICES[len(DEVICES)/2:]

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    gen_costs = []
    for i, device in enumerate(DEVICES_A):
        with tf.device(device):
            real_and_fake_data = tf.concat([
                all_real_data_splits[i], 
                all_real_data_splits[len(DEVICES_A)+i], 
                fake_data_splits[i], 
                fake_data_splits[len(DEVICES_A)+i]
            ], axis=0)
            real_and_fake_labels = tf.concat([
                labels_splits[i], 
                labels_splits[len(DEVICES_A)+i],
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i]
            ], axis=0)
            disc_all, disc_all_acgan, enc_all = Discriminator(real_and_fake_data, real_and_fake_labels)
            disc_real = disc_all[:BATCH_SIZE/len(DEVICES_A)]
            disc_fake = disc_all[BATCH_SIZE/len(DEVICES_A):]
            disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
            if CONDITIONAL and ACGAN:
                disc_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], labels=real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)])
                ))
                # disc_acgan_costs.append(tf.reduce_mean(
                #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan, labels=real_and_fake_labels)
                # ))
                disc_acgan_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], dimension=1)),
                            real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)]
                        ),
                        tf.float32
                    )
                ))
                disc_acgan_fake_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[BATCH_SIZE/len(DEVICES_A):], dimension=1)),
                            real_and_fake_labels[BATCH_SIZE/len(DEVICES_A):]
                        ),
                        tf.float32
                    )
                ))
            # Rita -- age
            enc_real = enc_all[:BATCH_SIZE/len(DEVICES_A)]
            enc_fake = enc_all[BATCH_SIZE/len(DEVICES_A):]
            disc_real_acgan = disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)]

            disc_real_acgan_i = tf.to_int32(tf.argmax(disc_real_acgan[:BATCH_SIZE/len(DEVICES_A)], dimension=1))

            gen_enc_real, _ = Generator(BATCH_SIZE/len(DEVICES_A), disc_real_acgan_i, noise=enc_real)
            # Match cost (cycle cost)
            match_real_cost = 10 * match_l1(gen_enc_real, real_and_fake_data[:BATCH_SIZE/len(DEVICES_A)])
            match_fake_cost = 10 * match_l2(enc_fake, tf.concat([noise_splits[i], noise_splits[len(DEVICES_A)+i]], axis=0))
            disc_costs.extend([match_real_cost, match_fake_cost])
            gen_costs.extend([match_real_cost, match_fake_cost])


    for i, device in enumerate(DEVICES_B): # Calcualte gradient penalty
        with tf.device(device):
            real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
            fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
            labels = tf.concat([
                labels_splits[i], 
                labels_splits[len(DEVICES_A)+i],
            ], axis=0)
            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES_A),1], 
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
            disc_costs.append(gradient_penalty)

    disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
    if CONDITIONAL and ACGAN:
        disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
        disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
        disc_cost = disc_wgan + (ACGAN_SCALE*disc_acgan)
    else:
        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan

    disc_params = lib.params_with_name('Discriminator.')

    if DECAY:
        decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.

    gen_acgan_costs = []
    for device in DEVICES:
        with tf.device(device):
            n_samples = BATCH_SIZE
            fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
            if CONDITIONAL and ACGAN:
                disc_fake, disc_fake_acgan, _ = Discriminator(Generator(n_samples,fake_labels)[0], fake_labels)
                gen_costs.append(-tf.reduce_mean(disc_fake))
                gen_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                ))
            else:
                gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels)[0], fake_labels)[0]))

            # match_fake_cost = 10 * match_l2()
    gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
    if CONDITIONAL and ACGAN:
        gen_cost += (ACGAN_SCALE_G*(tf.add_n(gen_acgan_costs) / len(DEVICES)))


    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Function for generating samples
    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples, _ = Generator(100, fixed_labels, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), Img_dir+'samples_{}.png'.format(frame)) 

    # Function for reconstruction samples
    recon_data_int = tf.placeholder(tf.int32, shape=[100, OUTPUT_DIM])
    recon_data = tf.reshape(2*((tf.cast(recon_data_int, tf.float32)/256.)-.5), [100, OUTPUT_DIM])
    recon_data += tf.random_uniform(shape=[100,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    _w, _ac, recon_z = Discriminator(recon_data, fixed_labels)
    recon_base, _ = Generator(100, tf.to_int32(tf.argmax(_ac, dimension=1)), noise=recon_z)

    def generate_recon_image(frame):
        recon_img = lib.cifar10.cifar_reconstructor_base(DATA_DIR)
        recon_base_, recon_z_ = session.run([recon_base, recon_z], feed_dict={recon_data_int: recon_img})
        recon_base_ = ((recon_base_+1.)*(255./2)).astype('int32')
        samples = np.vstack([recon_img[0], recon_base_[0]])
        for i in range(1, 100):
            samples = np.concatenate([samples, np.reshape(recon_img[i], (1, -1)), np.reshape(recon_base_[i], (1, -1))])
        lib.save_images.save_recon_images(samples.reshape((200, 3, 32, 32)), Img_dir+'recons_{}.png'.format(frame)) 

    interp_noise = tf.placeholder(tf.float32, shape=[110, 128])
    interp_label = tf.constant(np.repeat(np.array([0,1,2,3,4,5,6,7,8,9], dtype='int32'), 11))
    interp_samples, _ = Generator(110, interp_label, noise=interp_noise)
    # Function for interpolation
    def generate_interp_image_same_noise(frame):
        ends = np.random.randn(2, 128)
        m = ends[0] + np.reshape(np.tile(range(11), 128), (128, 11)).transpose() * (ends[1]-ends[0]) * 0.1
        noise = np.tile(m, (10, 1))
        print noise.shape
        samples = session.run(interp_samples, feed_dict={interp_noise:noise})
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_recon_images(samples.reshape((110, 3, 32, 32)), Img_dir+'interp_same_noise_{}.png'.format(frame)) 

    # Function for interpolation
    def generate_interp_image(frame):
        noise = np.zeros((1, 128))
        for n in xrange(10):
            ends = np.random.randn(2, 128)
            change_rate = ((ends[0]-ends[1])*0.1).reshape(1, 128)
            z = ends[1].reshape(1, 128)
            m = z
            for i in range(1,11):
                m = np.concatenate((m,z+change_rate*float(i)),axis=0)
            noise = np.concatenate([noise, m], axis=0)
        noise = noise[1:,]
        samples = session.run(interp_samples, feed_dict={interp_noise:noise})
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_recon_images(samples.reshape((110, 3, 32, 32)), Img_dir+'interp_{}.png'.format(frame)) 
        
    # Function for calculating inception score
    fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
    samples_100, _ = Generator(100, fake_labels_100)
    def get_inception_score(n):
        all_samples = []
        for i in xrange(n/100):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
        return lib.inception_score.get_inception_score(list(all_samples))

    train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR, Site_N, rate=Data_rate)
    def inf_train_gen():
        while True:
            for images,_labels in train_gen():
                yield images,_labels


    for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print "{} Params:".format(name)
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g == None:
                print "\t{} ({}) [no grad!]".format(v.name, shape_str)
            else:
                print "\t{} ({})".format(v.name, shape_str)
        print "Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        )

    saver = tf.train.Saver(max_to_keep=None)
    session.run(tf.initialize_all_variables())
    if load_dir is not None:
        saver.restore(session,load_dir)
        print 'Model restored'
    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        if iteration % PLOT_FREQUENCY == 0:
            start_time = time.time()

        for i in xrange(N_CRITIC):
            _data,_labels = gen.next()
            if CONDITIONAL and ACGAN:
                _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _, _enc_fake, _match_real, _match_fake = session.run([
                    disc_cost, disc_wgan, disc_acgan, 
                    disc_acgan_acc, disc_acgan_fake_acc, disc_train_op,
                    enc_fake, match_real_cost, match_fake_cost], 
                    feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

            else:
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

        _, _gen_cost = session.run([gen_train_op, gen_cost], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

        if iteration % PLOT_FREQUENCY == PLOT_FREQUENCY - 1:
            lib.plot.plot(Img_dir+'cost', _disc_cost)
            if CONDITIONAL and ACGAN:
                lib.plot.plot(Img_dir+'wgan_', _disc_wgan)
                lib.plot.plot(Img_dir+'acgan_', _disc_acgan)
                lib.plot.plot(Img_dir+'acc_real_', _disc_acgan_acc)
                lib.plot.plot(Img_dir+'acc_fake_', _disc_acgan_fake_acc)
            lib.plot.plot(Img_dir+'time_', time.time() - start_time)

        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY-1:
            inception_score = get_inception_score(50000)
            lib.plot.plot(Img_dir+'inception_50k', inception_score[0])
            lib.plot.plot(Img_dir+'inception_50k_std', inception_score[1])

        if iteration % MODEL_SAVE_FREQUENCY == MODEL_SAVE_FREQUENCY-1:
            saver.save(session,Model_dir+'model.ckpt',global_step=iteration)

        if iteration % 10 == 9:
            logging.info('Iteration %d  finished' % iteration) 
            print '\n----------Iteration %d ---------' % iteration
            print 'ecost', _disc_cost, 'wcost', _disc_wgan - _match_real - _match_fake, 'lcost', _disc_acgan
            print 'match_x',_match_real, 'match_z', _match_fake, 'enc_fake_var', np.mean(np.var(_enc_fake, axis=0))
            print 'gcost', _gen_cost - _match_real - _match_fake
        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            disc_acgan_accs_ = []
            for images,_labels in dev_gen():
                _dev_disc_cost,_disc_acgan_acc = session.run([disc_cost,disc_acgan_acc], feed_dict={all_real_data_int: images,all_real_labels:_labels})
                dev_disc_costs.append(_dev_disc_cost)
                disc_acgan_accs_.append(_disc_acgan_acc)
            lib.plot.plot(Img_dir+'dev_cost_', np.mean(dev_disc_costs))
            lib.plot.plot(Img_dir+'acgan_acc_', np.mean(disc_acgan_accs_))
            print 'dev_ac_acc:', np.mean(disc_acgan_accs_)
            generate_image(iteration, _data)
            generate_recon_image(iteration)
            generate_interp_image(iteration)
            generate_interp_image_same_noise(iteration)



        if (iteration < 500) or (iteration % 1000 == 999):
            lib.plot.flush()

        lib.plot.tick()