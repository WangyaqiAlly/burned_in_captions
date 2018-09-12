import time

import numpy
import tensorflow as tf

import os
import layers as L
import vat
import argparse
from read_celebA_3 import inputs, unlabeled_inputs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--label', help='label', default = None)
parser.add_argument('--ganlabel', help='label for GAN generation', default=None)
parser.add_argument('--transmit_only', help='use supervised model', default=False, action='store_true')
parser.add_argument('--all_local', help='train with all local data in all sites', default=False, action='store_true')
parser.add_argument('--percent_label', help='percentage of labelled data', default='0.5')
args = parser.parse_args()
print args

if args.label == None:
    print('ERROR: An label must be provided with --label.  Use None to skip')
    sys.exit(1)

if not os.path.exists('VAT_OUTPUT'):
    os.makedirs('VAT_OUTPUT')


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('device', '/gpu:0', "device")

tf.app.flags.DEFINE_string('dataset', 'celeba', "{cifar10, svhn, celeba}")

model_name = args.label
if args.ganlabel is not None:
    model_name = 'Generated_' + args.ganlabel + '-' + args.label + '-' + args.percent_label
if args.transmit_only:
    model_name = 'TransmittedOnly_' + args.label + '-' + args.percent_label
if args.all_local:
    model_name = 'AllLocal_' + args.label + '-' + args.percent_label

print('model name =', model_name)

tf.app.flags.DEFINE_string('log_dir', os.path.join('VAT_OUTPUT', model_name), "log_dir")
tf.app.flags.DEFINE_bool('validation', False, "")

tf.app.flags.DEFINE_integer('finetune_batch_size', 100, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('finetune_iter', 100, "the number of iteration for finetuning of BN stats")
tf.app.flags.DEFINE_integer('eval_batch_size', 500, "the number of examples in a batch")

# added by dtan:
tf.app.flags.DEFINE_integer('cls_num', 2, "the number of classes")
tf.app.flags.DEFINE_float('percent_label', float(args.percent_label), "the percent of data with label")

#if FLAGS.dataset == 'cifar10':
#    from cifar10 import inputs, unlabeled_inputs
#elif FLAGS.dataset == 'svhn':
#    from svhn import inputs, unlabeled_inputs 
#else: 
#    raise NotImplementedError


def build_finetune_graph(x):
    logit = vat.forward(x, is_training=True, update_batch_stats=True)
    with tf.control_dependencies([logit]):
        finetune_op = tf.no_op()
    return finetune_op


def build_eval_graph(x, y):
    logit = vat.forward(x, is_training=False, update_batch_stats=False)
    n_corrects = tf.cast(tf.equal(tf.argmax(logit, 1), tf.argmax(y,1)), tf.int32)
    return tf.reduce_sum(n_corrects), tf.shape(n_corrects)[0] 


def main(_):
    with tf.Graph().as_default() as g:
        with tf.device("/cpu:0"):
            label_src = 'transmit-' + args.percent_label
            if args.all_local:
                label_src = 'train-labelled'
            #images_eval_train, _ = inputs(args.label, args.ganlabel, batch_size=FLAGS.finetune_batch_size,
            #                              validation=FLAGS.validation,
            #                              shuffle=True)
            images_eval_test, labels_eval_test = inputs(args.label, args.ganlabel, batch_size=FLAGS.eval_batch_size,
                                                        train=False,
                                                        validation=FLAGS.validation,
                                                        shuffle=False, num_epochs=1, label_src = label_src)

        with tf.device(FLAGS.device):
            with tf.variable_scope("CNN") as scope:
                # Build graph of finetuning BN stats
                #finetune_op = build_finetune_graph(images_eval_train)
                #scope.reuse_variables()
                # Build eval graph
                n_correct, m = build_eval_graph(images_eval_test, labels_eval_test)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session()
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        print("Checkpoints:", ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(tf.local_variables_initializer()) 
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        #print("Finetuning...")
        #for _ in range(FLAGS.finetune_iter):
        #    sess.run(finetune_op)
            
        sum_correct_examples= 0
        sum_m = 0
        try:
            while not coord.should_stop():
                _n_correct, _m = sess.run([n_correct, m])
                sum_correct_examples += _n_correct
                sum_m += _m
        except tf.errors.OutOfRangeError:
            print('Done evaluation -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        print("Test: num_test_examples:{}, num_correct_examples:{}, accuracy:{}".format(
              sum_m, sum_correct_examples, sum_correct_examples/float(sum_m)))
   

if __name__ == "__main__":
    tf.app.run()