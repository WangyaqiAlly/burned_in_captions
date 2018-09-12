import time
from datetime import datetime
import numpy as np
import tensorflow as tf

import layers as L
import vat
import os

#import create_labeled_unlabeled_data_sets as speechInput
import readSpeechData as speechInput


import logging
logging.basicConfig()

logger = logging.getLogger()
logger.setLevel(logging.INFO)



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('device', '/gpu:0', "device")

tf.app.flags.DEFINE_string('dataset', 'cifar10', "{cifar10, lsun}")


tf.app.flags.DEFINE_integer('batch_size', 10, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('ul_batch_size', 10, "the number of unlabeled examples in a batch")
tf.app.flags.DEFINE_integer('eval_batch_size', 10, "the number of eval examples in a batch")
tf.app.flags.DEFINE_integer('eval_freq_batch', 500, "")
tf.app.flags.DEFINE_integer('report_freq_batch', 100, "")
tf.app.flags.DEFINE_integer('seed', 12345, "Seed")
tf.app.flags.DEFINE_integer('num_epochs', 200, "the number of epochs for training")
tf.app.flags.DEFINE_integer('epoch_decay_start', 80, "epoch of starting learning rate decay")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "initial leanring rate")
tf.app.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.app.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")
tf.app.flags.DEFINE_string('gantype', 'CTGAN_SEMI',
                           'where to store the dataset')
tf.app.flags.DEFINE_string('site', 'central','which site')
tf.app.flags.DEFINE_string('method', 'vatent', "{vat, vatent, baseline}")

tf.app.flags.DEFINE_integer('num_labeled', 2100, "The number of labeled examples")
tf.app.flags.DEFINE_integer('num_unlabeled', 18000, "The number of labeled examples")
tf.app.flags.DEFINE_integer('num_test', 2100, "The number of test labeled examples")
tf.app.flags.DEFINE_string('log_dir', "./Record/GenData_10PercentLabeled/site"+FLAGS.site+"_"+str(FLAGS.num_labeled)+'_'+str(FLAGS. num_unlabeled)+'/', "log_dir")
tf.app.flags.DEFINE_bool('is_load_pretrained_model',False,"")
tf.app.flags.DEFINE_string('pretrained_dir', "./Record/GenData_10PercentLabeled/site/02", "pretrained_dir")
tf.app.flags.DEFINE_integer('cls_num', 10, "number of classes")

#if FLAGS.dataset == 'cifar10':
#    from cifar_inputs import inputs, unlabeled_inputs
#elif FLAGS.dataset == 'lsun':
#    from lsun_input import inputs, unlabeled_inputs
#else: 
#    raise NotImplementedError


#NUM_EVAL_EXAMPLES = 1000

NUM_EVAL_EXAMPLES_TR = 2100
NUM_EVAL_EXAMPLES_TS = 2100
max_test_acc =0.0
NUM_ITER_PER_EPOCH = int(FLAGS.num_unlabeled/FLAGS.ul_batch_size)


print(FLAGS.__dict__['__flags'])


def build_training_graph(x, y, ul_x, lr, mom):
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False,
    )
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


def main(_):
    
    print(FLAGS.epsilon, FLAGS.top_bn)
    np.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(np.random.randint(1234))
    with tf.Graph().as_default() as g:
        if not FLAGS.log_dir:
            logdir = None
            writer_train = None
            writer_test = None
        else:
            logdir = FLAGS.log_dir
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            fh = logging.FileHandler(logdir + '{}.log'.format(datetime.now().strftime('%Y%m%d-%H%M%S')))
            logger.addHandler(fh)

            logger.info(FLAGS.__dict__['__flags'])

            writer_train = tf.summary.FileWriter(logdir + "/train", g)
            writer_test = tf.summary.FileWriter(logdir + "/test", g)

        #with tf.device("/cpu:0"):
        if True:
	    ##trainLabeledData, trainUnlabeledData, testData = speechInput.readSpeechImages()
	    images, labels, ul_images, test_images, test_labels = speechInput.readSpeechImages()
	    #print "size of trainLabeledData, trainUnlabeledData, testData", len(trainLabeledData), len(trainUnlabeledData), len(testData)
	    images = images[0:FLAGS.num_labeled]
	    labels = labels[0:FLAGS.num_labeled]
	    ul_images = ul_images[0:FLAGS.num_unlabeled]
	    test_images = test_images[0:FLAGS.num_test]
	    test_labels = test_labels[0:FLAGS.num_test]
	    '''images, labels = tf.train.shuffle_batch_join(
            			trainLabeledData,
            			batch_size=FLAGS.batch_size,
            			capacity=FLAGS.num_labeled+ 3 * FLAGS.batch_size,
            			min_after_dequeue=FLAGS.num_labeled)
            ul_images, _ = tf.train.shuffle_batch_join(
            			trainUnlabeledData,
            			batch_size=FLAGS.ul_batch_size,
            			capacity=FLAGS.num_unlabeled + 3*FLAGS.ul_batch_size,
            			min_after_dequeue=FLAGS.num_unlabeled)
            
            images_eval_train, labels_eval_train = tf.train.shuffle_batch_join(
            						trainLabeledData,
            						batch_size=FLAGS.eval_batch_size,
            						capacity=FLAGS.num_labeled+ 3 * FLAGS.eval_batch_size,
            						min_after_dequeue=FLAGS.num_labeled)
            						
            ul_images_eval_train, _ = tf.train.shuffle_batch_join(
            				trainUnlabeledData,
            				batch_size=FLAGS.eval_batch_size,
            				capacity=FLAGS.num_unlabeled + 3*FLAGS.eval_batch_size,
            				min_after_dequeue=FLAGS.num_unlabeled)
            				
            images_eval_test, labels_eval_test = tf.train.shuffle_batch_join(
            			testData,
            			batch_size=FLAGS.eval_batch_size,
            			capacity=FLAGS.num_labeled+ 3 * FLAGS.eval_batch_size,
            			min_after_dequeue=FLAGS.num_labeled)'''
            			
            #print "images.shape {0} labels.shape {1} ul_images.shape {2} test_images.shape {3} test_labels.shape {4}".format(images.shape,labels.shape,ul_images.shape,images_eval_test.shape,labels_eval_test.shape)		
            			
            #images, labels = inputs(batch_size=FLAGS.batch_size,
            #                        train=True,
            #                        shuffle=True)
            #ul_images,_ = unlabeled_inputs(batch_size=FLAGS.ul_batch_size,
            #                             shuffle=True)
            #images_eval_train, labels_eval_train = inputs(batch_size=FLAGS.eval_batch_size,
            #                                              train=True,
            #                                              shuffle=True)
            #ul_images_eval_train,_ = unlabeled_inputs(batch_size=FLAGS.eval_batch_size,
            #                                        shuffle=True)

            #images_eval_test, labels_eval_test = inputs(batch_size=FLAGS.eval_batch_size,
            #                                            train=False,
            #                                            shuffle=True)

        #with tf.device(FLAGS.device):
        if True:
            lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            mom = tf.placeholder(tf.float32, shape=[], name="momentum")
            with tf.variable_scope("CNN") as scope:
            	x = tf.placeholder('float32', [None,128,128,1],name='x'); print x
    		y = tf.placeholder('int32',[None, 10]); print y
    		ul_x = tf.placeholder('float32', [None,128,128,1],name='ul_x'); print ul_x
                # Build training graph
                loss, train_op, global_step = build_training_graph(x, y, ul_x, lr, mom)
                scope.reuse_variables()
                # Build eval graph
                losses_eval_train = build_eval_graph(x, y, ul_x)
                losses_eval_test = build_eval_graph(x, y, ul_x)

            init_op = tf.global_variables_initializer()


        def load_pretrain(sess):
            if FLAGS.is_load_pretrained_model:
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrained_dir))
            else:
                return init_op

        saver = tf.train.Saver(tf.global_variables())
        sv = tf.train.Supervisor(
            is_chief=True,
            logdir=logdir,
            #init_op=init_op,
            init_fn=load_pretrain,
            init_feed_dict={lr: FLAGS.learning_rate, mom: FLAGS.mom1},
            saver=saver,
            global_step=global_step,
            summary_op=None,
            summary_writer=None,
            save_model_secs=1800, recovery_wait_secs=0)

        print("Training...")
        max_test_acc=0.0
        with sv.managed_session() as sess:
        #with tf.Session() as sess:
            for ep in range(FLAGS.num_epochs):
                if sv.should_stop():
                    break

                if ep < FLAGS.epoch_decay_start:
                    #feed_dict = {lr: FLAGS.learning_rate, mom: FLAGS.mom1}
                    myLR = FLAGS.learning_rate
                    myMOM = FLAGS.mom1
                else:
                    decayed_lr = ((FLAGS.num_epochs - ep) / float(
                        FLAGS.num_epochs - FLAGS.epoch_decay_start)) * FLAGS.learning_rate
                    myLR = decayed_lr
                    myMOM = FLAGS.mom2
                    #feed_dict = {lr: decayed_lr, mom: FLAGS.mom2}

                sum_loss = 0
                start = time.time()
                for iter in range(NUM_ITER_PER_EPOCH-1):
                    ### First shuffle all the data
                    ind = np.arange(FLAGS.num_labeled)
                    np.random.shuffle(ind)
                    images = images[ind]
                    labels = labels[ind]
                    ind = np.arange(FLAGS.num_unlabeled)
                    np.random.shuffle(ind)
                    ul_images = ul_images[ind]
                    ind = np.arange(FLAGS.num_test)
                    np.random.shuffle(ind)
                    test_images = test_images[ind]
                    test_labels = test_labels[ind]
                    
                    ### Then choose batch_size of labeled data for each iteration randomly
		    myInd = np.random.choice(FLAGS.num_labeled,FLAGS.batch_size)
		    x_input = images[myInd]
		    y_input = labels[myInd]
		    ul_x_input = ul_images[iter*FLAGS.ul_batch_size:(iter+1)*FLAGS.ul_batch_size]
                    _, batch_loss, current_global_step = sess.run([train_op, loss, global_step],
                                                feed_dict={x: x_input, y: y_input, ul_x: ul_x_input, lr: myLR, mom: myMOM})
                    sum_loss += batch_loss
                    if (iter + 1) % FLAGS.report_freq_batch == 0:
                        logger.info("Epoch: {},batch: {}, CE_loss_train:{}".format(ep,iter, sum_loss / iter))

                    if (iter + 1) % FLAGS.eval_freq_batch== 0 or iter + 1 == NUM_ITER_PER_EPOCH:
                        # Eval on training data
                        act_values_dict = {}
                        for key, _ in losses_eval_train.iteritems():
                            act_values_dict[key] = 0
                        n_iter_per_epoch = NUM_EVAL_EXAMPLES_TR / FLAGS.eval_batch_size
                        for i in range(n_iter_per_epoch-1):
                            ind = np.random.choice(FLAGS.num_unlabeled,FLAGS.eval_batch_size)
                            values = losses_eval_train.values()
                            act_values = sess.run(values, feed_dict={x: images[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], y: labels[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], ul_x: ul_images[ind]})
                            for key, value in zip(act_values_dict.keys(), act_values):
                                act_values_dict[key] += value
                        summary = tf.Summary()
                        for key, value in act_values_dict.iteritems():
                            logger.info("train-{}:{}".format(key, value / n_iter_per_epoch))
                            summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
                        if writer_train is not None:
                            writer_train.add_summary(summary, current_global_step)

                        # Eval on test data
                        act_values_dict = {}
                        for key, _ in losses_eval_test.iteritems():
                            act_values_dict[key] = 0
                        n_iter_per_epoch = NUM_EVAL_EXAMPLES_TS / FLAGS.eval_batch_size
                        for i in range(n_iter_per_epoch-1):
                            ind = np.random.choice(FLAGS.num_unlabeled,FLAGS.eval_batch_size)
                            values = losses_eval_test.values()
                            act_values = sess.run(values, feed_dict={x: test_images[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], y: test_labels[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], ul_x: ul_images[ind]})
                            for key, value in zip(act_values_dict.keys(), act_values):
                                act_values_dict[key] += value
                        summary = tf.Summary()
                        for key, value in act_values_dict.iteritems():
                            value = value / n_iter_per_epoch
                            logger.info("test-{}:{}".format(key, value))
                            summary.value.add(tag=key, simple_value=value)
                        if writer_test is not None:
                            writer_test.add_summary(summary, current_global_step)
                        test_acc = act_values_dict['Acc'] / n_iter_per_epoch
                        if test_acc > max_test_acc:
                            max_test_acc_iter = ep
                            max_test_acc = test_acc
                            saver.save(sess, sv.save_path + '-best', global_step=global_step)
                        logger.info("-----l:{} ul:{} site:{}  max_test_acc:{},epoch:{}------------".format(
                            FLAGS.num_labeled, FLAGS.num_unlabeled, FLAGS.site,
                            max_test_acc, max_test_acc_iter))
                end = time.time()
                logger.info("Epoch: {}, CE_loss_train:{} , elapsed_time: {}".format(ep,sum_loss /NUM_ITER_PER_EPOCH, end - start))

             
            saver.save(sess, sv.save_path, global_step=global_step)
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
