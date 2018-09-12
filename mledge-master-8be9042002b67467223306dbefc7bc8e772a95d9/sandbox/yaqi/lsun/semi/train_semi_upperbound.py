import time
from datetime import datetime
import numpy
import tensorflow as tf

import layers as L
import vat
import os



import logging
logging.basicConfig()

logger = logging.getLogger()
logger.setLevel(logging.INFO)



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('device', '/gpu:0', "device")

tf.app.flags.DEFINE_string('dataset', 'lsun', "{cifar10, lsun}")
tf.app.flags.DEFINE_integer('cls_num', 20,'how many classes {20,10}')


tf.app.flags.DEFINE_integer('seed', 1, "initial random seed")
tf.app.flags.DEFINE_bool('validation',False, "")

tf.app.flags.DEFINE_integer('batch_size', 64, "the number of examples in a batch")
#tf.app.flags.DEFINE_integer('batch_size', 16, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('ul_batch_size', 256, "the number of unlabeled examples in a batch")
#tf.app.flags.DEFINE_integer('ul_batch_size', 256, "the number of unlabeled examples in a batch")
tf.app.flags.DEFINE_integer('eval_batch_size', 500, "the number of eval examples in a batch")
tf.app.flags.DEFINE_integer('eval_freq', 2000, "")
tf.app.flags.DEFINE_integer('iter_report_freq', 200, "")
tf.app.flags.DEFINE_integer('num_epochs', 200, "the number of epochs for training")
tf.app.flags.DEFINE_integer('epoch_decay_start', 80, "epoch of starting learning rate decay")
#tf.app.flags.DEFINE_integer('num_iter_per_epoch', 400, "the number of updates per epoch")
tf.app.flags.DEFINE_float('learning_rate', 0.002, "initial leanring rate")
tf.app.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.app.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")
#tf.app.flags.DEFINE_string('gantype', 'CTGAN_UNCOND_ZCA',
#                           'where to store the dataset')
tf.app.flags.DEFINE_string('unlabeled_type', 'real','which site')
tf.app.flags.DEFINE_integer('site_num', 1,'which site')
#tf.app.flags.DEFINE_floa('site_num', 1,'which site')
tf.app.flags.DEFINE_string('method', 'vatent', "{vat, vatent, baseline}")

tf.app.flags.DEFINE_integer('train_size_per_site', 600000, "The number of training examples{5037800,6241830 6847288}")
tf.app.flags.DEFINE_integer('test_size', 120000, "The number of validation examples")

tf.app.flags.DEFINE_string('log_dir', "./Record/upperbound/"+'/sitenum_'+ str(FLAGS.site_num) +'/', "log_dir")
tf.app.flags.DEFINE_bool('is_load_pretrained_model',True,"")
tf.app.flags.DEFINE_string('pretrained_dir', "./Record/upperbound/"+'/sitenum_'+ str(FLAGS.site_num-1)+"/", "pretrained_dir")


if FLAGS.dataset == 'cifar10':
    from four_sites_input import inputs, unlabeled_inputs
elif FLAGS.dataset == 'lsun':
    from lsun_input_upperbound import inputs, unlabeled_inputs
else: 
    raise NotImplementedError


#NUM_EVAL_EXAMPLES = 1000

NUM_EVAL_EXAMPLES_TR = 5000
NUM_EVAL_EXAMPLES_TS = 5000
max_test_acc =0.0



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
    numpy.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(numpy.random.randint(1234))
    with tf.Graph().as_default() as g:
        with tf.device("/cpu:0"):
            images, labels = inputs(batch_size=FLAGS.batch_size,
                                    train=True,
                                    shuffle=True)
            ul_images,_ = unlabeled_inputs(batch_size=FLAGS.ul_batch_size,
                                         shuffle=True)

            images_eval_train, labels_eval_train = inputs(batch_size=FLAGS.eval_batch_size,
                                                          train=True,
                                                          #validation=FLAGS.validation,
                                                          shuffle=True)
            ul_images_eval_train,_ = unlabeled_inputs(batch_size=FLAGS.eval_batch_size,
                                                    #validation=FLAGS.validation,
                                                    shuffle=True)

            images_eval_test, labels_eval_test = inputs(batch_size=FLAGS.eval_batch_size,
                                                        train=False,
                                                        #validation=FLAGS.validation,
                                                        shuffle=True)

        with tf.device(FLAGS.device):
            lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            mom = tf.placeholder(tf.float32, shape=[], name="momentum")
            with tf.variable_scope("CNN") as scope:
                # Build training graph
                loss, train_op, global_step = build_training_graph(images, labels, ul_images, lr, mom)
                scope.reuse_variables()
                # Build eval graph
                losses_eval_train = build_eval_graph(images_eval_train, labels_eval_train, ul_images_eval_train)
                losses_eval_test = build_eval_graph(images_eval_test, labels_eval_test, images_eval_test)

            init_op = tf.global_variables_initializer()


        NUM_ITER_PER_EPOCH = int(FLAGS.train_size_per_site * FLAGS.site_num / FLAGS.batch_size)
        print ('NUM_ITER_PER_EPOCH:', NUM_ITER_PER_EPOCH)

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

        saver = tf.train.Saver(tf.global_variables())

        def load_pretrain(sess):
            if FLAGS.is_load_pretrained_model:
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrained_dir))
            else:
                return init_op

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
            for ep in range(FLAGS.num_epochs):
                if sv.should_stop():
                    break

                if ep < FLAGS.epoch_decay_start:
                    feed_dict = {lr: FLAGS.learning_rate, mom: FLAGS.mom1}
                else:
                    decayed_lr = ((FLAGS.num_epochs - ep) / float(
                        FLAGS.num_epochs - FLAGS.epoch_decay_start)) * FLAGS.learning_rate
                    feed_dict = {lr: decayed_lr, mom: FLAGS.mom2}

                sum_loss = 0
                start = time.time()
                for i in range(NUM_ITER_PER_EPOCH):
                    _, batch_loss, _ = sess.run([train_op, loss, global_step],
                                                feed_dict=feed_dict)
                    sum_loss += batch_loss

                    if (i + 1) % FLAGS.iter_report_freq == 0:
                        logger.info("Epoch:{} Iter: {}, CE_loss_train:{} ".format(ep, i, batch_loss))

                    if (i + 1) % FLAGS.eval_freq == 0 or (i + 1) == NUM_ITER_PER_EPOCH:
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
                            max_test_acc_iter = i
                            max_test_acc = test_acc
                        logger.info("-----site_num:{} type:{}, max_test_acc:{},iter:{}------------".format(
                                 FLAGS.site_num, FLAGS.unlabeled_type,max_test_acc,max_test_acc_iter))
                end = time.time()
                logger.info("Epoch: {} finish, CE_loss_train:{} , elapsed_time: {}".format(ep,
                                                                                               sum_loss / NUM_ITER_PER_EPOCH,
                                                                                               end - start))


                    #logger.info("----gantype:{} site:{} max_test_acc:{},iter:{}------------".format(FLAGS.unlabeled_inputs_dir,FLAGS.site,max_test_acc_iter, max_test_acc))
            saver.save(sess, sv.save_path, global_step=global_step)
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
