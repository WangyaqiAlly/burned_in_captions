from resnet import *
import tensorflow as tf
import os

MOMENTUM = 0.9
import logging
logging.basicConfig()
import time
from datetime import datetime
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './Record/classifier_v0/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('eval_size', 640, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')
tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 60000, '''At which step to decay the learning rate''')


def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size

def train( train_images, train_labels,test_images,test_labels):
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    logdir = FLAGS.train_dir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    fh = logging.FileHandler(logdir + '/{}.log'.format(datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger.addHandler(fh)
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    is_training = tf.placeholder(dtype=tf.bool, shape=None)

    logits = inference(train_images,
                       num_classes=FLAGS.cls_num,
                       is_training=is_training,
                       bottleneck=True,
                       num_blocks=[3, 4, 6, 3])  # resnet-50

    loss_ = loss(logits, train_labels)
    predictions = tf.nn.softmax(logits)

    top1_error = top_k_error(predictions, train_labels, 1)

    logits_test = inference(test_images,
                       num_classes=FLAGS.cls_num,
                       is_training=is_training,
                       bottleneck=True,
                       num_blocks=[3, 4, 6, 3])  # resnet-50
    predictions_test = tf.nn.softmax(logits_test)
    pred_bin_test = tf.argmax(logits_test,1)
    with tf.name_scope("evaluate_metric"):
        precision, precision_update = tf.metrics.precision(test_labels,pred_bin_test)
        recall,recall_update = tf.metrics.recall(test_labels,pred_bin_test)
        accuracy,accuracy_update = tf.metrics.accuracy(test_labels,pred_bin_test)
    evaluate_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,scope="evaluate_metric")

    top1_error_test = top_k_error(predictions_test, test_labels, 1)


    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    lr = tf.placeholder(dtype=tf.float32, shape=[])
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.summary.merge_all()

    # init = tf.initialize_all_variables()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    evaluate_vars_initializer = tf.variables_initializer(var_list=evaluate_vars)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            tf.logging.info( "No checkpoint to continue from in {}".format(FLAGS.train_dir))
            sys.exit(1)
        tf.logging.info( "resume {}".format(latest))
        saver.restore(sess, latest)

    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        i = [train_op, loss_]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)


        if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
            FLAGS.learning_rate = 0.1 *  FLAGS.learning_rate
            print 'Learning rate decayed to ', FLAGS.init_lr

        o = sess.run(i, { is_training: True,lr:FLAGS.learning_rate })

        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = 'step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f}sec/batch)'
            tf.logging.info(format_str.format(step, loss_value, examples_per_sec, duration))

        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

        # Run validation periodically
        if step == 1: #and step % 100 == 0:
            top1_error_values=0.0
            for i in xrange(int(FLAGS.eval_size / FLAGS.batch_size)):
                _, top1_error_value = sess.run([val_op, top1_error], { is_training: False })
                top1_error_values += top1_error_value
            acc_val = 1.0 - float(top1_error_values/i)
            tf.logging.info('Validation accuracy{:.3f}'.format(acc_val))
        if step >1: #and step % 100 == 0:
            tf.logging.info("evaluate on test set....")
            sess.run(evaluate_vars_initializer)
            _top1_error_tests =0.0
            for i in xrange(int(FLAGS.test_size/FLAGS.batch_size)):
                    _,_,_,_top1_error_test = sess.run([precision_update,recall_update,accuracy_update, top1_error_test], {is_training: False})
                    _top1_error_tests += _top1_error_test

            _precision,_recall,_accuracy = sess.run([precision,recall,accuracy], {is_training: False})
            _acc_test =1.0- float(_top1_error_tests) /i
            tf.logging.info('Test accuracy:{:.3f} Presicion:{:.3f} recall:{:.3}f, acc_notformtf:{:.3f}'.format(_accuracy,_precision,_recall,_acc_test))



