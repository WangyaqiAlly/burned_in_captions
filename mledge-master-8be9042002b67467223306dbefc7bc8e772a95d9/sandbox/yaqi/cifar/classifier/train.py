import time
from datetime import datetime
import numpy
import numpy as np
import tensorflow as tf

import layers as L
import cnn
import os



import logging
logging.basicConfig()

logger = logging.getLogger()
logger.setLevel(logging.INFO)



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('device', '/gpu:0', "device")

tf.app.flags.DEFINE_string('dataset', 'cifar10', "{semigan_aclabel, semigan_oraclelabel,cifar10}")



tf.app.flags.DEFINE_integer('batch_size', 512, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('eval_batch_size', 512, "the number of eval examples in a batch")
tf.app.flags.DEFINE_integer('eval_freq_batch', 1000, "")
tf.app.flags.DEFINE_integer('report_freq_batch', 100, "")
tf.app.flags.DEFINE_integer('num_epochs', 200, "the number of epochs for training")
tf.app.flags.DEFINE_integer('epoch_decay_start', 160, "epoch of starting learning rate decay")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "initial leanring rate")
tf.app.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.app.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")
tf.app.flags.DEFINE_string('gantype', 'CTGAN_SEMI','which site')
tf.app.flags.DEFINE_integer('num_labeled_examples', 500000, "The number of labeled examples")
#tf.app.flags.DEFINE_integer('num_unlabeled', 100000, "The number of labeled examples")
#tf.app.flags.DEFINE_integer('dataset_seed', 10, "dataset seed")
#tf.app.flags.DEFINE_string('log_dir', "./Record/2sites_new/site:"+FLAGS.site+'/'+FLAGS.unlabeled_inputs_dir+'/', "log_dir")
tf.app.flags.DEFINE_string('log_dir', "./Record/multisites_site0_1250/",'')
if FLAGS.dataset == 'semigan_aclabel':
    from semigan_ac_input import inputs
elif FLAGS.dataset == 'semigan_oraclelabel':
    from semigan_oracle_input import inputs
elif FLAGS.dataset == 'cifar10':
    from cifar_input import inputs

else:
    raise NotImplementedError


#NUM_EVAL_EXAMPLES = 1000

NUM_EVAL_EXAMPLES_TR = 5000
NUM_EVAL_EXAMPLES_TS = 10000
max_test_acc =0.0
NUM_ITER_PER_EPOCH = int(FLAGS.num_labeled_examples/FLAGS.batch_size)


print(FLAGS.__dict__['__flags'])


def build_training_graph(x, y,lr, mom):
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False,
    )
    logit = cnn.forward(x)
    loss = L.ce_loss(logit, y)
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
    tvars = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(loss, tvars)
    train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    return loss, train_op, global_step


def build_eval_graph(x, y):
    losses = {}
    logit = cnn.forward(x, is_training=False, update_batch_stats=False)
    nll_loss = L.ce_loss(logit, y)
    losses['NLL'] = nll_loss
    acc = L.accuracy(logit, y)
    losses['Acc'] = acc
    return losses



def main(_):
    print(FLAGS.epsilon, FLAGS.top_bn)
    numpy.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(numpy.random.randint(1234))
    # ZCA_components = np.load('../Data/4sites_nozca/cifar_nozca/zca_cifar/components_cifar_3072.npy')
    # ZCA_mean = np.load('../Data/4sites_nozca/cifar_nozca/zca_cifar/mean_cifar_3072.npy')
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
        with tf.device("/cpu:0"):
            # ZCA_components_T_tf = tf.convert_to_tensor(ZCA_components.T, np.float32)
            # ZCA_mean_tf = tf.convert_to_tensor(ZCA_mean, np.float32)

            images, labels = inputs(batch_size=FLAGS.batch_size,
                                    train=True,
                                    shuffle=True)

            images_eval_train, labels_eval_train = inputs(batch_size=FLAGS.eval_batch_size,
                                                          train=True,
                                                          #validation=FLAGS.validation,
                                                          shuffle=True)

            images_eval_test, labels_eval_test = inputs(batch_size=FLAGS.eval_batch_size,
                                                        train=False,
                                                        #validation=FLAGS.validation,
                                                        shuffle=True)



            # def zca(images):
            #     images = tf.reshape(images, [-1, 32 * 32 * 3])
            #     images = tf.matmul(images - ZCA_mean_tf, ZCA_components_T_tf)
            #     images = tf.reshape(images, [-1, 32 , 32 , 3])
            #     return images

            # images=zca(images)
            # ul_images=zca(ul_images)
            # images_eval_train=zca(images_eval_train)
            # ul_images_eval_train=zca(ul_images_eval_train)
            # images_eval_test=zca(images_eval_test)


        with tf.device(FLAGS.device):
            lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            mom = tf.placeholder(tf.float32, shape=[], name="momentum")
            with tf.variable_scope("CNN") as scope:
                # Build training graph
                loss, train_op, global_step = build_training_graph(images, labels, lr, mom)
                scope.reuse_variables()
                # Build eval graph
                losses_eval_train = build_eval_graph(images_eval_train, labels_eval_train)
                losses_eval_test = build_eval_graph(images_eval_test, labels_eval_test)

            init_op = tf.global_variables_initializer()



        saver = tf.train.Saver(tf.global_variables())

        def load_pretrain(sess):
            if FLAGS.pretrained_dir:
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrained_dir))
            else:
                tf.global_variables_initializer()


        sv = tf.train.Supervisor(
            is_chief=True,
            logdir=logdir,
            init_op=init_op,
            #init_fn=load_pretrain,
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
                for iter in range(NUM_ITER_PER_EPOCH):
                    _, batch_loss, current_global_step = sess.run([train_op, loss, global_step],
                                                                  feed_dict=feed_dict)
                    sum_loss += batch_loss
                    if (iter + 1) % FLAGS.report_freq_batch == 0:
                        logger.info("Epoch: {},batch: {}, CE_loss_train:{}".format(ep, iter, sum_loss / iter))

                    if (iter + 1) % FLAGS.eval_freq_batch == 0 or iter + 1 == NUM_ITER_PER_EPOCH:
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
                        logger.info("-----label type :{}, l:{}, max_test_acc:{},epoch:{}------------".format(FLAGS.dataset,
                            FLAGS.num_labeled_examples,
                            max_test_acc, max_test_acc_iter))

                        # logger.info("----gantype:{} site:{} max_test_acc:{},iter:{}------------".format(FLAGS.unlabeled_inputs_dir,FLAGS.site,max_test_acc_iter, max_test_acc))
                end = time.time()
                logger.info(
                    "Epoch: {}, CE_loss_train:{} , elapsed_time: {}".format(ep, sum_loss / NUM_ITER_PER_EPOCH,
                                                                            end - start))

            saver.save(sess, sv.save_path, global_step=global_step)
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
