from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
import cv2
print 'cv2 ' + cv2.__version__
import numpy as np
print 'numpy ' + np.__version__

parser = argparse.ArgumentParser()
# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--resnet_size', type=int, default=32,
                    help='The size of the ResNet model to use.')

parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)

parser.add_argument(
    '--data_format', type=str, default='channels_last',
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

args = parser.parse_args()


tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]

if not os.path.exists(tag):
    os.makedirs(tag)



_HEIGHT = 64
_WIDTH = 64
_DEPTH = 3
_NUM_CLASSES = 2
_NUM_DATA_FILES = 5

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9


_NUM_IMAGES = {
    'train': 1000000,
    'test': 14000,
}



_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

TrainFolder1 = '../Record/SmilingPeople/generatedImages/'
#TrainFolder1 = '../Record/SmilingPeople/RealImages/trainData/' 
TestFolder1 = '../Record/SmilingPeople/RealImages/testData/'
TrainFolder0 = '../Record/NonSmilingPeople/generatedImages/'
#TrainFolder0 = '../Record/NonSmilingPeople/RealImages/trainData/'
TestFolder0 = '../Record/NonSmilingPeople/RealImages/testData/'

folders = [TrainFolder1, TrainFolder0, TestFolder1, TestFolder0]
trainData = []
testData = []
trainLabels = []
testLabels = []
### Reading Data
for i,folder in enumerate(folders):
  fileNames = os.listdir(folder)
  for f in fileNames:
	#print folder,f
	img = cv2.imread(folder+f,cv2.IMREAD_COLOR)
	img = cv2.resize(img,(_WIDTH, _HEIGHT), interpolation = cv2.INTER_CUBIC)
	img = img.transpose(2,0,1)
	if i<2:
		trainData.append(img)
		tmp = np.zeros(_NUM_CLASSES)
		if i==0:
			tmp[1] += 1
			trainLabels.append(tmp)
		else:
			tmp[0] += 1
			trainLabels.append(tmp)
	else:
		testData.append(img)
		tmp = np.zeros(_NUM_CLASSES)
		if i==2:
			tmp[1] += 1
			testLabels.append(tmp)
		else:
			tmp[0] += 1
			testLabels.append(tmp)
	
trainData = np.array(trainData[:_NUM_IMAGES['train']]).reshape(_NUM_IMAGES['train'],_WIDTH,_HEIGHT,3).astype('float32')
trainData = trainData/255.

testData = np.array(testData).reshape(_NUM_IMAGES['test'],_WIDTH,_HEIGHT,3).astype('float32')
testData = testData/255.

trainLabels = np.array(trainLabels[:_NUM_IMAGES['train']]).reshape(_NUM_IMAGES['train'],_NUM_CLASSES).astype('int32')
testLabels = np.array(testLabels).reshape(_NUM_IMAGES['test'],_NUM_CLASSES).astype('int32')

print "trainData.shape", trainData.shape, 'trainData.max',trainData.max(), 'trainLabels.shape', trainLabels.shape
print "testData.shape", testData.shape, 'testData.max',testData.max(), 'testLabels.shape', testLabels.shape




def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   data_format):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

  return tf.identity(inputs, name)


def resnet_v2(inputs,is_training=False):
  """Generator for CIFAR-10 ResNet v2 models.

  Args:
    resnet_size: A single integer for the size of the ResNet model.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.

  Raises:
    ValueError: If `resnet_size` is invalid.
  """
  resnet_size = args.resnet_size
  num_classes = _NUM_CLASSES
  data_format=args.data_format

  if resnet_size % 6 != 2:
    raise ValueError('resnet_size must be 6n + 2:', resnet_size)

  num_blocks = (resnet_size - 2) // 6

  with tf.variable_scope('C',reuse=False):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=16, kernel_size=3, strides=1,
        data_format=data_format); print inputs.shape
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = block_layer(
        inputs=inputs, filters=16, block_fn=building_block, blocks=num_blocks,
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format); print inputs.shape
    inputs = block_layer(
        inputs=inputs, filters=32, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format); print inputs.shape
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format); print inputs.shape

    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format); print inputs.shape

    inputs = batch_norm_relu(inputs, is_training, data_format); print inputs.shape
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=8, strides=1, padding='VALID',
        data_format=data_format); print inputs.shape
    
    inputs = tf.identity(inputs, 'final_avg_pool'); print inputs.shape
    inputs = tf.reshape(inputs, [-1, 128])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
  return inputs




#tf.summary.image('images', features, max_outputs=6)
x = tf.placeholder('float32', [None,_HEIGHT, _WIDTH, _DEPTH],name='x') ; print x
y = tf.placeholder('int32', [None,_NUM_CLASSES],name='y') ; print y

logits = resnet_v2(x,is_training=True)
predictions = { 'classes': tf.argmax(logits, axis=1),
      		'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
}

# Calculate loss, which includes softmax cross entropy and L2 regularization.
cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=y)

# Create a tensor named cross_entropy for logging purposes.
tf.identity(cross_entropy, name='cross_entropy')
tf.summary.scalar('cross_entropy', cross_entropy)

# Add weight decay to the loss.
loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

params={'resnet_size': args.resnet_size,
          'data_format': args.data_format,
          'batch_size': args.batch_size}
#if mode == tf.estimator.ModeKeys.TRAIN:
if True:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 128, the learning rate should be 0.1.
    initial_learning_rate = 0.1 * params['batch_size'] / 128
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
else:
    train_op = None

#myLogits = resnet_v2(x)
#myPredictions = { 'classes': tf.argmax(myLogits, axis=1),
#      		'probabilities': tf.nn.softmax(myLogits, name='softmax_tensor')
#}
#accuracy = tf.metrics.accuracy(
#      tf.argmax(labels, axis=1), myPredictions['classes'])
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), predictions['classes']), tf.float32))
#accuracy = tf.metrics.accuracy(
#      tf.argmax(y, axis=1), predictions['classes'])

#metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
#tf.identity(accuracy[1], name='train_accuracy')
#tf.summary.scalar('train_accuracy', accuracy[1])

#dloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=class_out,labels=label)); print dloss

'''dopt = tf.train.AdamOptimizer(learning_rate=args.lr)
v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'C')
dgrads = dopt.compute_gradients(loss,var_list=v1)
dtrain = dopt.apply_gradients(dgrads)
dnorm = tf.global_norm([i[0] for i in dgrads])

# Batch norm requires update ops to be added as a dependency to the train_op
#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_ops):
#     train_op = optimizer.minimize(loss, global_step)


accuracy = tf.metrics.accuracy(
      tf.argmax(y, axis=1), predictions['classes'])
metrics = {'accuracy': accuracy}

# Create a tensor named train_accuracy for logging purposes
tf.identity(accuracy[1], name='train_accuracy')
tf.summary.scalar('train_accuracy', accuracy[1])'''


saver = tf.train.Saver(max_to_keep = 10000)
step = 0
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(args.train_epochs):
    	# shuffle both the data and the labels
		ind = np.arange(trainData.shape[0])
		np.random.shuffle(ind)
		trainData = trainData[ind]
		trainLabels = trainLabels[ind]
		dc=0.
		dn=0.
		acc = 0.
		t=0.
		for j in range(0,trainData.shape[0],args.batch_size):
			input_x = trainData[j:j+args.batch_size]
			input_y = trainLabels[j:j+args.batch_size]
			_,acc_ = sess.run([train_op, accuracy], feed_dict={x:input_x,y:input_y})
			#dc += dc_
			#dn += dn_
			acc += acc_
			t += 1.
			step += 1
			if (step % 5000) == 0:
				savepath = saver.save(sess, os.path.join(tag, 'model'), global_step=step, write_meta_graph=False)
				print 'saving ',savepath
			#if j%500==0:
		trainSetAcc = acc/t
		#dcSummary = dc/t

		t = 0.
		testAccRes = 0.
		tmpAcc = 0
		for k in range(0,testData.shape[0],args.batch_size):
			input_x_test = testData[k:k+args.batch_size]
			input_y_test = testLabels[k:k+args.batch_size]
			tmpAcc = sess.run(accuracy, feed_dict={x:input_x_test, y:input_y_test})
			#print tmpAcc, input_y_test[100]
			testAccRes += tmpAcc
			t += 1.
		testAccRes = testAccRes/t
		#print 'epoch',i,'Training cost',dcSummary , 'Train Set Accuracy:', trainSetAcc , 'Test Set Accuracy:', testAccRes
		print 'epoch',i, 'Train Set Accuracy:', trainSetAcc , 'Test Set Accuracy:', testAccRes


