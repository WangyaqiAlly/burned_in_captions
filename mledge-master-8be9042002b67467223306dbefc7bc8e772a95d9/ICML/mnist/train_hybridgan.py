'''
    train_hybridgan.py

    training of HybridGan scheme at the central site
'''

import os
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

import layers as L
import vat

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

'convert to argparse'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('device', '/gpu:0', "device")

tf.app.flags.DEFINE_integer('seed', 1, "initial random seed")
tf.app.flags.DEFINE_bool('validation',False, "")

tf.app.flags.DEFINE_integer('batch_size', 64, "the number of examples in a batch")
# tf.app.flags.DEFINE_integer('batch_size', 32, "the number of examples in a batch")
#tf.app.flags.DEFINE_integer('batch_size', 16, "the number of examples in a batch")
# tf.app.flags.DEFINE_integer('ul_batch_size', 128, "the number of unlabeled examples in a batch")
tf.app.flags.DEFINE_integer('ul_batch_size', 256, "the number of unlabeled examples in a batch")

tf.app.flags.DEFINE_integer('eval_batch_size', 100, "the number of eval examples in a batch")
tf.app.flags.DEFINE_integer('eval_freq', 1, "how often to evaluate trained model")
tf.app.flags.DEFINE_integer('eval_freq_batch', 100, "evaluation interval in # of batch, within each epoch")

tf.app.flags.DEFINE_integer('epoch_decay_start', 100, "epoch of starting learning rate decay")
tf.app.flags.DEFINE_integer('num_iter_per_epoch', 400, "the number of updates per epoch")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "initial leanring rate")
tf.app.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.app.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")
# tf.app.flags.DEFINE_string('site', 'central','which site')
tf.app.flags.DEFINE_string('method', 'vatent', "{vat, vatent, baseline}")

tf.app.flags.DEFINE_bool('is_load_pretrained_model',False,"whether to load pretrain model")
tf.app.flags.DEFINE_string('pretrained_dir', "./Record/pretrained", "pretrained_dir")


NUM_CLASSES = 10
NUM_EXAMPLES_TEST = 10000

NUM_EVAL_EXAMPLES_TR = 10000
NUM_EVAL_EXAMPLES_TS = 10000
max_test_acc =0.0

print(FLAGS.__dict__['__flags'])

def generate_batch(example, data_size, batch_size, shuffle):
    
    num_preprocess_threads = 10

    if shuffle:
        ret = tf.train.shuffle_batch(example,
            			     batch_size=batch_size,
            			     num_threads=num_preprocess_threads,
            			     capacity=data_size+ 3 * batch_size,
            			     min_after_dequeue=data_size)
    else:
        ret = tf.train.batch(example,
            		     batch_size=batch_size,
            		     num_threads=num_preprocess_threads,
            		     allow_smaller_final_batch=True,
            		     capacity=data_size)
    return ret

def generate_batch_join(example, dataset_size, batch_size, shuffle):

    if shuffle:
        ret = tf.train.shuffle_batch_join(example,
            				  batch_size=batch_size,
            				  capacity=dataset_size+ 3 * batch_size,
			            	  min_after_dequeue=dataset_size)
    else:
        ret = tf.train.batch(example,
            		     batch_size=batch_size,
                             allow_smaller_final_batch=False,
                             capacity=dataset_size+ 3 * batch_size)
    return ret

def generate_filename_queue(filenames, data_dir, num_epochs=None):

    print "filenames in queue:"
    for i in range(len(filenames)):
        filenames[i] = os.path.join(data_dir, filenames[i])
	print filenames[i]

    return tf.train.string_input_producer(filenames, num_epochs=num_epochs)


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
    image = tf.reshape(image, [28, 28, 1])
    label = tf.one_hot(tf.cast(features['label'], tf.int32), 10)
    return image, label


def load_test_labeled(args, batch_size=100, num_epochs=None):

    print 'load testing dataset'
    filenames = ['test.tfrecords']
    filename_queue = generate_filename_queue(filenames, args.dbdir, num_epochs)
    image, label = read(filename_queue, False)
    tf.logging.info('test, labeled | building queue from:{}....'.format(filenames))
    tf.logging.info('image shape {},labels shape {}'.format(image.shape, label.shape))
    tf.logging.info('max pixel {}'.format(np.amax(image)))
    
    num_examples = NUM_EXAMPLES_TEST
    return generate_batch([image, label], num_examples, batch_size, False)

def load_train_labeled(args, batch_size=100, num_epochs=None):

    print 'load training dataset, labeled'
    # filenames = ['train.tfrecords']
    # nfile = len(filenames)

    nfile = args.nfile
    filenames = []
    for i in range(nfile):
	filenames.append('sample-%03d-of-%03d-pct%02d.tfrecords' % (i, args.nsite, args.pct)) 

    filename_queue = generate_filename_queue(filenames, args.lddir, num_epochs)
    example_list = [read(filename_queue, True) for _ in range(nfile)]

    tf.logging.info('train, labeled | building queue from:{}....'.format(filenames))
    tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
    tf.logging.info('max pixel {}'.format(np.amax(example_list[0][0])))
    num_examples =  args.nlabeled  
        
    return generate_batch_join(example=example_list, 
                               dataset_size=num_examples, 
                               batch_size=batch_size,
                               shuffle=True)

def load_train_unlabeled(args, batch_size=100, num_epochs=None):
    
    print 'load training dataset, unlabeled'

    if args.original: 
	'use original as unlabeled'
	nfile = 1
        filenames = ['train.tfrecords']
        filename_queue = generate_filename_queue(filenames, args.dbdir, num_epochs)
        example_list = [read(filename_queue, True) for _ in range(nfile)]
    else: 
        nsite = args.nsite
        nfile = args.nfile
        filenames = []
	if args.uncond:
	    'use unconditionally generated as unlabeled'
            for i in range(nfile):
	        filenames.append('gen-%03d-of-%03d.tfrecords' % (i, nsite)) 
	        filenames.append('sample-%03d-of-%03d-pct%02d.tfrecords' % (i, nsite, args.pct))
        else: 
	    'use conditionally generated as unlabeled'
            for i in range(nfile):
	        filenames.append('gen-%03d-of-%03d-semi.tfrecords' % (i, nsite)) 
	        filenames.append('sample-%03d-of-%03d-pct%02d.tfrecords' % (i, nsite, args.pct))
        filename_queue = generate_filename_queue(filenames, args.uddir, num_epochs)
        example_list = [read(filename_queue, True) for _ in range(nfile*2)]

    tf.logging.info('train, unlabeled | building queue from:{}....'.format(filenames))
    tf.logging.info('image shape {},labels shape {}'.format(example_list[0][0].shape, example_list[0][1].shape))
    tf.logging.info('max pixel {}'.format(np.amax(example_list[0][0])))

    num_examples= args.nunlabeled 
    return generate_batch_join(example=example_list, 
                               dataset_size=num_examples, 
                               batch_size=batch_size,
                               shuffle=True)

def build_training_graph(x, y, ul_x, lr, mom):
    
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False,)

    logit = vat.forward(x)
    nll_loss = L.ce_loss(logit, y)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        if FLAGS.method == 'vat':
            ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
            vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit)
            additional_loss = vat_loss
        elif FLAGS.method == 'vatent':
            ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
            vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit)
            ent_loss = L.entropy_y_x(ul_logit)
            additional_loss = vat_loss + ent_loss
        elif FLAGS.method == 'baseline':
            additional_loss = 0
        else:
            raise NotImplementedError
        loss = nll_loss + additional_loss

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
    tvars = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(loss, tvars)
    train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    return loss, train_op, global_step


def build_eval_graph(x, y, ul_x):
    losses = {}
    logit = vat.forward(x, is_training=False, update_batch_stats=False)
    nll_loss = L.ce_loss(logit, y)
    losses['NLL'] = nll_loss
    acc = L.accuracy(logit, y)
    losses['Acc'] = acc
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    at_loss = vat.adversarial_loss(x, y, nll_loss, is_training=False)
    losses['AT_loss'] = at_loss
    ul_logit = vat.forward(ul_x, is_training=False, update_batch_stats=False)
    vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit, is_training=False)
    losses['VAT_loss'] = vat_loss
    return losses

def load_pretrain(sess):
    if FLAGS.is_load_pretrained_model:
	print 'loading pretrained model'
	saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrained_dir))
    else:
	print 'loading init_op' 
        return init_op

def main(_):

    print '*** input arguments ***'
    print args

#    print(FLAGS.epsilon, FLAGS.top_bn)
    np.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(np.random.randint(1234))

    with tf.Graph().as_default() as g:
        if args.nolog:
            logdir = None
            writer_train = None
            writer_test = None
        else:
            logdir = args.logdir # FLAGS.log_dir
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            fh = logging.FileHandler(logdir + '{}.log'.format(datetime.now().strftime('%Y%m%d-%H%M%S')))
            logger.addHandler(fh)
            logger.info(FLAGS.__dict__['__flags'])

            writer_train = tf.summary.FileWriter(logdir + "/train", g)
            writer_test = tf.summary.FileWriter(logdir + "/test", g)

        with tf.device("/cpu:0"):
	    print '*** gathering images and labels ***'
            images, labels = load_train_labeled(args, batch_size=FLAGS.batch_size)
            ul_images, _   = load_train_unlabeled(args, batch_size=FLAGS.ul_batch_size)

	    print 'train | labeled images and labels: ', images.shape, labels.shape	
	    print 'train | unlabeled images: ', ul_images.shape
	
#            images_eval_train, labels_eval_train = load_train_labeled(args, batch_size=FLAGS.eval_batch_size)
#            ul_images_eval_train, _   = load_train_unlabeled(args, batch_size=FLAGS.eval_batch_size)

#	    print 'eval | labeled images and labels: ', images.shape, labels.shape	
#	    print 'eval | unlabeled images: ', ul_images.shape

            images_eval_test, labels_eval_test = load_test_labeled(args, batch_size=FLAGS.eval_batch_size)

	    print 'test | images and labels: ', images_eval_test.shape, labels_eval_test.shape
	
        with tf.device(FLAGS.device):
            lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            mom = tf.placeholder(tf.float32, shape=[], name="momentum")

	    print 'build training & eval graphs'
            with tf.variable_scope("CNN") as scope:

                # Build training graph
                loss, train_op, global_step = build_training_graph(images, labels, ul_images, lr, mom)
                scope.reuse_variables()

                # Build eval graph
                # losses_eval_train = build_eval_graph(images_eval_train, 
		# 				     labels_eval_train, 
		# 				     ul_images_eval_train)

                losses_eval_test = build_eval_graph(images_eval_test, 
						    labels_eval_test, 
						    images_eval_test)

            init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.global_variables())
        sv = tf.train.Supervisor(
            is_chief=True,
            logdir=logdir,
            init_feed_dict={lr: FLAGS.learning_rate, mom: FLAGS.mom1},
            saver=saver,
            global_step=global_step,
            summary_op=None,
            summary_writer=None,
            save_model_secs=1800, recovery_wait_secs=0)

        print  "Training..."
        max_test_acc=0.0
	niter = args.nunlabeled/FLAGS.ul_batch_size
        with sv.managed_session() as sess:
            for ep in range(args.nepoch):

	        print 'epoch %d ...' % ep
                if sv.should_stop():
                    break

		'update learning rate & momentum'
                if ep < FLAGS.epoch_decay_start:
                    feed_dict = {lr: FLAGS.learning_rate, mom: FLAGS.mom1}
                else:
                    decayed_lr = (args.nepoch - ep)/float(args.nepoch - FLAGS.epoch_decay_start)
		    decayed_lr *= FLAGS.learning_rate
                    feed_dict = {lr: decayed_lr, mom: FLAGS.mom2}

                sum_loss = 0
                start = time.time()
#                for iter in range(FLAGS.num_iter_per_epoch):
                for iter in xrange(niter):
                    _, batch_loss, _ = sess.run([train_op, loss, global_step],
                                                feed_dict=feed_dict)
                    sum_loss += batch_loss
                    if (iter + 1) % FLAGS.eval_freq_batch == 0:
                        logger.info("Epoch: {},batch: {}, CE_loss_train:{}".format(ep,iter, sum_loss / iter))

                end = time.time()
		avgloss = sum_loss / FLAGS.num_iter_per_epoch
		telapsed = end-start
                logger.info("Epoch: {}, CE_loss_train:{} , elapsed_time: {}".format(ep, avgloss, telapsed))

                if (ep + 1) % FLAGS.eval_freq == 0 or ep + 1 == args.nepoch:
		    '''
                    # Eval on training data
                    act_values_dict = {}
                    for key, _ in losses_eval_train.iteritems():
                        act_values_dict[key] = 0
                    n_iter_per_epoch = NUM_EVAL_EXAMPLES_TR / FLAGS.eval_batch_size
                    for i in range(n_iter_per_epoch):
                        values = losses_eval_train.values()
                        act_values = sess.run(values)
                        for key, value in zip(act_values_dict.keys(), act_values):
                            act_values_dict[key] += value
                    summary = tf.Summary()
                    current_global_step = sess.run(global_step)
                    for key, value in act_values_dict.iteritems():
                        logger.info("train-{}:{}".format(key, value / n_iter_per_epoch))
                        summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
                    if writer_train is not None:
                        writer_train.add_summary(summary, current_global_step)
		    '''

                    # Eval on test data
                    act_values_dict = {}
                    for key, _ in losses_eval_test.iteritems():
                        act_values_dict[key] = 0
                    n_iter_per_epoch = NUM_EVAL_EXAMPLES_TS / FLAGS.eval_batch_size
                    for i in range(n_iter_per_epoch):
                        values = losses_eval_test.values()
                        act_values = sess.run(values)
                        for key, value in zip(act_values_dict.keys(), act_values):
                            act_values_dict[key] += value
                    summary = tf.Summary()
                    current_global_step = sess.run(global_step)
                    for key, value in act_values_dict.iteritems():
                        value=value / n_iter_per_epoch
                        logger.info("test-{}:{}".format(key, value ))
                        summary.value.add(tag=key, simple_value=value)
                    if writer_test is not None:
                        writer_test.add_summary(summary, current_global_step)
                    test_acc = act_values_dict['Acc'] / n_iter_per_epoch
                    if test_acc > max_test_acc:
                        max_test_acc_iter = ep
                        max_test_acc = test_acc

		    logmsg = "--- epoch {} | l:{} ul:{} nfile: {}, nsite:{}, curr_acc:{}, max_acc:{},at_epoch:{} ---".format(
				ep, 
				args.nlabeled, 
				args.nunlabeled,
				args.nfile,  
				args.nsite, 
				test_acc, 
				max_test_acc,
				max_test_acc_iter)
		    logger.info(logmsg)

            saver.save(sess, sv.save_path, global_step=global_step)
        
        sv.stop()

if __name__ == "__main__":
   
     # --------- start of main procedures -----------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
    parser.add_argument('--logdir', help='directory of logging info', default='./record/')
    parser.add_argument('--dbdir', help='directory of original dataset for testing',
	   			   default='/home/xiaoqzhu/dataset/mnist/')
    parser.add_argument('--lddir', help='directory of labeled dataset',
	   			   default='/home/xiaoqzhu/dataset/mnist/multi_site')
    parser.add_argument('--uddir', help='directory of unlabeled dataset',
	   			   default='/home/xiaoqzhu/dataset/mnist/multi_site')
 
    parser.add_argument('--nsite', help='number of distributed sites', default=10, type=int)
    parser.add_argument('--nfile', help='number of contributing sites', default=10, type=int)
    parser.add_argument('--nlabeled', help='number of labeled samples', default=600, type=int)
    parser.add_argument('--nunlabeled', help='number of unlabeled samples', default=600000, type=int)
    parser.add_argument('--nepoch', help='number of epochs', default=100, type=int)
    parser.add_argument('--nolog', default=False, action='store_true')
    parser.add_argument('--original', default=False, action='store_true')
    parser.add_argument('--uncond', default=False, action='store_true')
    args = parser.parse_args()

    args.pct = args.nlabeled*100/60000
    t = (args.logdir, args.nfile, args.nsite, args.pct)
    if args.original: 
 	'using originals as unlabeled'
        args.logdir = '%s/original_%03d_of_%03d_pct%03d/' % t 
	args.nunlabeled = 60000
    elif args.uncond: 
 	'using generated/synthetic as unlabeled'
        args.logdir = '%s/synthetic_%03d_of_%03d_pct%03d/' % t 
    else: 
	'using conditionally generated as unlabeled'
        args.logdir = '%s/syn_cond_%03d_of_%03d_pct%03d/' % t 

    print args
    
    tf.app.run()
