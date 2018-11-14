

import tensorflow as tf
import os, sys, pickle
import numpy as np
# from scipy import linalg

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('aug_trans',   True, "")
tf.app.flags.DEFINE_bool('aug_flip', True, "")



def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_images_and_labels(images, labels, filepath):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)
    for index in range(num_examples):
        image = images[index].tolist()
        image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(32),
            'width': _int64_feature(32),
            'depth': _int64_feature(3),
            'label': _int64_feature(int(labels[index])),
            'image': image_feature}))
        writer.write(example.SerializeToString())
    writer.close()


def read_old(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([3072], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    image = features['image']
    image = tf.reshape(image, [FLAGS.width, FLAGS.height, 3])
    label = tf.one_hot(tf.cast(features['label'], tf.int32), FLAGS.cls_num)
    return image, label


def read(filename_queue,train,generated,thread_id):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'depth': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'language': tf.FixedLenFeature([], tf.int64),
                                           'nlines': tf.FixedLenFeature([], tf.int64),
                                           'bbx': tf.VarLenFeature(tf.float32),
                                           'image': tf.FixedLenFeature([], tf.string)})

    image = tf.decode_raw(features['image'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    # nlines = tf.cast(features['nlines'], tf.int32)
    # bbox = tf.cast(features['bbx'], tf.float32)
    language = tf.cast(features['language'], tf.int32)
    channel = 3

    # image = tf.reshape(image, [FLAGS., height, channel])

    image = tf.reshape(image, [width, height, 3])
    #label = tf.one_hot(tf.cast(features['label'], tf.int32), 20)
    #print('label before transform',label.shape)
    # image = transform(image) if train else image
    if FLAGS.network == 'resnet':
        image = image_preprocessing(image, [], train, thread_id)
    # image = image /255.0 - 0.5
    else:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, (FLAGS.img_height, FLAGS.img_width),
                                         align_corners=True)
        image = tf.squeeze(image, [0])
        return image

    label = label #tf.one_hot(tf.cast(features['label'], tf.int32),  FLAGS.cls_num)
    return image, label   #,language

def generate_batch(
        example,
        dataset_size,
        batch_size, shuffle):
    """
    Arg:
        list of tensors.
    """
    num_preprocess_threads = 10
    if shuffle:
        ret = tf.train.shuffle_batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=dataset_size + 3 * batch_size,
            min_after_dequeue=dataset_size)

    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=True,
            capacity=dataset_size + 3 * batch_size
        )

    return ret


def generate_batch_join(
        example,
        dataset_size,
        batch_size, shuffle):
    """
    Arg:
        list of tensors.
    """

    if shuffle:
        ret = tf.train.shuffle_batch_join(
            example,
            batch_size=batch_size,
            capacity=dataset_size+ 3 * batch_size,
            min_after_dequeue=dataset_size )
    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            allow_smaller_final_batch=True,
            capacity=dataset_size+ 3 * batch_size)

    return ret



def transform(image):
    image = tf.reshape(image, [FLAGS.width, FLAGS.height, 3])
    if FLAGS.aug_trans or FLAGS.aug_flip:
        print("augmentation")
        if FLAGS.aug_trans:
            # image = tf.pad(image, [[2, 2], [2, 2], [0, 0]])
            image = tf.random_crop(image, [FLAGS.width, FLAGS.height, 3])
        if FLAGS.aug_flip:
            image = tf.image.random_flip_left_right(image)
    return image


def generate_filename_queue(filenames, data_dir, num_epochs=None):
    print("filenames in queue:", filenames)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(data_dir, filenames[i])
    return tf.train.string_input_producer(filenames, num_epochs=num_epochs)



def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
    # with  tf.name_scope([image], scope, 'distort_color'):
    color_ordering = thread_id % 2

    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def distort_image(image, height, width, bbox, thread_id=0, scope=None):
    """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
    # NOTE(ry) I unceremoniously removed all the bounding box code.
    # Original here: https://github.com/tensorflow/models/blob/148a15fb043dacdd1595eb4c5267705fbd362c6a/inception/inception/image_processing.py

    distorted_image = image

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    resize_method = thread_id % 4
    distorted_image = tf.image.resize_images(distorted_image,(height,
                                             width), resize_method)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if not thread_id:
        tf.summary.image('cropped_resized_image',
                         tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # # Randomly distort the colors.
    distorted_image = distort_color(distorted_image, thread_id)

    if not thread_id:
        tf.summary.image('final_distorted_image',
                         tf.expand_dims(distorted_image, 0))
    return distorted_image


def eval_image(image, height, width, scope=None):
    """Prepare one image for evaluation.

  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
    # with tf.name_scope([image, FLAGS.height, FLAGS.width], scope, 'eval_image'):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        # image = tf.image.central_crop(image, central_fraction=0.875)
        # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, (height, width),
                                     align_corners=True)
    image = tf.squeeze(image, [0])
    return image


def image_preprocessing(image, bbox, train, thread_id=0):
    """Decode and preprocess one image for evaluation or training.

  Args:
    image_buffer: JPEG encoded string Tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    train: boolean
    thread_id: integer indicating preprocessing thread

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
    if bbox is None:
        raise ValueError('Please supply a bounding box.')
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if train:
        image = distort_image(image, FLAGS.img_height, FLAGS.img_width, bbox, thread_id)
    else:
        image = eval_image(image, FLAGS.img_height, FLAGS.img_width)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image