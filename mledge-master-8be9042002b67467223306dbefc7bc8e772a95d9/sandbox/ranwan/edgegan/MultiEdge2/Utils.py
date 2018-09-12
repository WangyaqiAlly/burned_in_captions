import cPickle
import tensorflow as tf
import os
import struct
import numpy as np
import cv2
import logging
import sys
import datetime
class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, distort):
        # type: (object, object) -> object
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        result_images = self._images[start:end]
        temp = self._labels[start:end]
        # result_labels = np.zeros([batch_size, 10])
        # result_labels[np.arange(batch_size), temp] = 1
        result_labels = temp
        return result_images, result_labels
# The graph_dir needs to be the directory includes *ubyte
def load_mnist(graph_dir='.', flag='train'):
    logger = logging.getLogger('load mnist')
    if flag == 'train':
        # Load mnist train images
        with open(graph_dir + '/train-images-idx3-ubyte', 'rb') as f:
            h = struct.unpack('>IIII', f.read(16))
            data = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1], h[2], h[3], 1)).astype('float32')
            data = data / 255. - 0.5
        # Load mnist train label
        with open(graph_dir + '/train-labels-idx1-ubyte', 'rb') as f:
            h = struct.unpack('>II', f.read(8))
            label = np.fromstring(f.read(), dtype=np.uint8).astype('int32')
        logger.debug('data.shape {:s} data.min() {:f}, data.max() {:f} label.shape {:s}'
                     .format((data.shape,), data.min(), data.max(), (label.shape,)))
        return data, label
    else:  # Load test data
        with open(graph_dir + '/t10k-images-idx3-ubyte', 'rb') as f:
            h = struct.unpack('>IIII', f.read(16))
            data = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1], h[2], h[3], 1)).astype('float32')
            data = data / 255. - 0.5
        with open(graph_dir + '/t10k-labels-idx1-ubyte', 'rb') as f:
            h = struct.unpack('>II', f.read(8))
            label = np.fromstring(f.read(), dtype=np.uint8).astype('int32')
        logger.debug('data.shape {:s} data.min() {:f}, data.max() {:f} label.shape {:s}'
                     .format((data.shape,), data.min(), data.max(), (label.shape,)))
        return data, label
# graph_dir is the root directory of
def load_imageNet(graph_dir, flag='train', edge_id='0', time_id=1):
    logger = logging.getLogger('load imageNet')
    if flag == 'train':
        logger.info('Dealing with training data')
        num_images = 0
        parent_path = os.path.join(graph_dir, 'train', 'site-%s' % edge_id, 'time-%02d' % time_id)
        num_classes = len(os.listdir(parent_path))
        for cls in range(1, num_classes + 1):
            graph_path = os.path.join(parent_path, 'class-%02d' % cls)
            num_images += len(os.listdir(graph_path))
        images = np.zeros((num_images, 32, 32, 3))
        labels = np.zeros(num_images, dtype=np.int)
        index = 0
        for cls in range(1, num_classes + 1):
            graph_path = os.path.join(parent_path, 'class-%02d' % cls)
            for image_name in os.listdir(graph_path):
                image = cv2.imread(os.path.join(graph_path, image_name))
                label = cls - 1
                assert (label >= 0)
                images[index] = image / 255.0 - 0.5
                labels[index] = label
                index += 1
        logger.info('Finished fold %d with %d samples.' % (time_id, index))
    else:
        logger.info('Dealing with validation data:')
        image_num = 0
        parent_path = os.path.join(graph_dir, 'validation')
        num_classes = len(os.listdir(parent_path))
        for cls in range(1, num_classes + 1):
            path = os.path.join(parent_path, 'class-%02d' % cls)
            image_num += len(os.listdir(path))
        images = np.zeros((image_num, 32, 32, 3))
        labels = np.zeros(image_num, dtype=np.int)
        index = 0
        for c in range(1, num_classes + 1):
            path = os.path.join(parent_path, 'class-%02d' % c)
            for filename in os.listdir(path):
                image = cv2.imread(os.path.join(path, filename))
                label = c - 1
                assert (label >= 0)
                images[index] = image / 255.0 - 0.5
                labels[index] = label
                index += 1
        logger.info('Finished test set with %d samples.' % index)
    return images, labels
def load_cifar10(graph_dir, flag='train', edge_id='1', time_id=1):
    logger = logging.getLogger('Load Cifar10')
    images, labels = None, None
    if flag == 'train':
        logger.info('Dealing with training data')
        parent_path = os.path.join(graph_dir, 'train', 'site-%s' % edge_id, 'time-%02d' % time_id)
        fileNames = os.listdir(parent_path)
        print fileNames
    else:
        parent_path = os.path.join(graph_dir, 'validation')
        fileNames = os.listdir(parent_path)
        print fileNames
    for f in fileNames:
        with open(parent_path + '/' + f, 'rb') as fo:
            tempDict = cPickle.load(fo)
        tempArr = tempDict['data']
        tempLabel = tempDict['labels']
        logger.debug('len of tempLabel: %d  type of tempArr: %s' % (len(tempLabel), (tempArr.shape,)))
        per_images = np.zeros((len(tempLabel), 32, 32, 3))
        per_labels = np.zeros(len(tempLabel), dtype=np.int)
        for i in range(len(tempLabel)):
            im = np.reshape(tempArr[i], (32, 32, 3), order='F')
            rows, cols, _ = im.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
            im = cv2.warpAffine(im, M, (cols, rows))
            per_images[i] = im / 255.0 - 0.5
            per_labels[i] = tempLabel[i]
        if images is None:
            images, labels = per_images, per_labels
        else:
            images = np.concatenate([images, per_images], axis=0)
            labels = np.concatenate([labels, per_labels], axis=0)
    if images is not None:
        images = images[...,[2,1,0]]
    logger.info('Finished fold %d with %s samples.' % (time_id, labels.shape))
    #visualization_color((images+0.5)*255., labels, edge_id)
    return images, labels
# ########
# Used for visualization
#       The input images of this function should be an array with shape (n, dim, dim, channel)
#       with element scale (0, 255)
# ########
def visualization_grey(images, labels, edge_id='', save_img=False, img_dir='', imgs=None):
    samples_per_class = 3
    generated_label = labels
    classes = np.unique(generated_label)
    # For each class, choose `samples_per_class` images to show
    for _class in classes:
        one_class_all_ids = np.where(generated_label == _class)[0]
        if len(one_class_all_ids) != 0:
            one_class_num_ids = np.random.choice(one_class_all_ids, samples_per_class)  # .tolist()
            if imgs is None:
                imgs = np.concatenate(images[one_class_num_ids], axis=0)
            else:
                imgs = np.concatenate((imgs, np.concatenate(images[one_class_num_ids], axis=0)), axis=1)
    # Generate the image, append the label, and show
    img = cv2.cvtColor(cv2.resize(imgs.astype('uint8'), (len(classes) * 100, samples_per_class * 100)),
                       cv2.COLOR_GRAY2RGB)
    for i, k in enumerate(classes):
        cv2.putText(img, "{:d}".format(k), (i * 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow('Generated images' + edge_id, img)
    if save_img:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = img_dir + '/' + edge_id + '_{:%Y.%m.%d.%H.%M.%S.%f}.png'.format(datetime.datetime.now())
        cv2.imwrite(img_path, img)
    cv2.waitKey(10)
# ########
# Used for visualization colorful images
#       The input images of this function should be an array with shape (n, dim, dim, channel)
#       with element scale (0, 255)
# ########
def visualization_color(images, labels, edge_id='', save_img=False, img_dir='', imgs=None):
    samples_per_class = 10
    generated_label = labels
    classes = np.unique(generated_label)
    # For each class, choose `samples_per_class` images to show
    for _class in classes:
        one_class_all_ids = np.where(generated_label == _class)[0]
        if len(one_class_all_ids) != 0:
            one_class_num_ids = np.random.choice(one_class_all_ids, samples_per_class)  # .tolist()
            if imgs is None:
                imgs = np.concatenate(images[one_class_num_ids], axis=0)
            else:
                imgs = np.concatenate((imgs, np.concatenate(images[one_class_num_ids], axis=0)), axis=1)
    # Generate the image, append the label, and show
    img = imgs.astype('uint8')
    width = img.shape[0]
    length = img.shape[1]
    img = cv2.resize(img, (length * 3, width * 3), interpolation=cv2.INTER_CUBIC)
    for i, k in enumerate(classes):
        cv2.putText(img, "{:d}".format(k), (i * 32 * 3, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow('Generated images' + edge_id, img)
    if save_img:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = img_dir + '/' + edge_id + '_{:%Y.%m.%d.%H.%M.%S.%f}.png'.format(datetime.datetime.now())
        cv2.imwrite(img_path, img)
    cv2.waitKey(10)
def update_progress(epoch, progress):
    barLength = 50  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Epoch %s Done...\r\n" % epoch
    block = int(round(barLength * progress))
    text = "\rEpoch: {0} Percent: [{1}] {2}% {3}".format(epoch, "#" * block + "-" * (barLength - block), progress * 100,
                                                         status)
    sys.stdout.write(text)
    sys.stdout.flush()
# ################
# Load images to scale (-0.5, 0.5)
# ################
class ImageLoader(object):
    def __init__(self, image_type, graph_dir, edge_id='0', data_start_time=1, data_end_time=1):
        self.graph_dir = graph_dir
        self.logger = logging.getLogger('Image Loader')
        self.logger.debug(
            '\n~~~~~In IMAGELOADER: ~~~~~\n image_type: {:s} \n graph_dir: {:s}'.format(image_type, graph_dir))
        train_images, train_labels, test_images, test_labels = None, None, None, None
        # Load data
        if image_type == 'mnist':
            for time in xrange(data_start_time, data_end_time + 1):
                train_images_t, train_labels_t = load_mnist(os.path.join(graph_dir, 'train'))
                if train_images is None:
                    train_images, train_labels = train_images_t, train_labels_t
                else:
                    train_images = np.concatenate([train_images, train_images_t], axis=0)
                    train_labels = np.concatenate([train_labels, train_labels_t], axis=0)
            test_images, test_labels = load_mnist(os.path.join(graph_dir, 'validation'), 'test')
        elif image_type == 'imageNet':
            for time in xrange(data_start_time, data_end_time + 1):
                train_images_t, train_labels_t = load_imageNet(graph_dir, 'train', edge_id, time)
                if train_images is None:
                    train_images, train_labels = train_images_t, train_labels_t
                else:
                    train_images = np.concatenate([train_images, train_images_t], axis=0)
                    train_labels = np.concatenate([train_labels, train_labels_t], axis=0)
            test_images, test_labels = load_imageNet(graph_dir, 'validation')
        elif image_type == 'cifar10':
            for time in xrange(data_start_time, data_end_time + 1):
                train_images_t, train_labels_t = load_cifar10(graph_dir, 'train', edge_id, time)
                if train_images is None:
                    train_images, train_labels = train_images_t, train_labels_t
                else:
                    train_images = np.concatenate([train_images, train_images_t], axis=0)
                    train_labels = np.concatenate([train_labels, train_labels_t], axis=0)
            test_images, test_labels = load_cifar10(graph_dir, 'validation')
        else:
            train_images, train_labels, test_images, test_labels = None, None, None, None
            self.logger.error('Not valid image type')
        # Wrap as Dataset
        if train_images is not None and len(train_images) > 0:
            self.train = DataSet(train_images, train_labels)
            self.logger.info('Train dataset shape: {:s} with min {:.2f}, max {:.2f}'
                             .format(self.train.images.shape,
                                     self.train.images.min(), self.train.images.max()))
        if test_images is not None and len(test_images) > 0:
            self.test = DataSet(test_images, test_labels)
            self.logger.info('Test dataset shape: {:s} with min {:.2f}, max {:.2f}'
                             .format(self.test.images.shape,
                                     self.test.images.min(), self.test.images.max()))
    def update_images(self):
        pass
class ImageGenerator(object):
    def __init__(self, num_images, image_type,
                 graph_dir, record_dir_sets, edge_ids):
        self.num_images = num_images
        self.record_dir_sets = record_dir_sets
        self.edge_ids = edge_ids
        self.logger = logging.getLogger('Image Generator')
        test_dir = os.path.join(graph_dir, 'validation/')
        self.logger.debug('\n\n ~~~~~~~~~~~~~~~~~In IMAGEGENERATOR: ~~~~~~~~~~~~~~~')
        self.logger.debug(' image_type: {:s} \n test_dir: {:s}'
                          .format(image_type, test_dir))
        # Load test images and label
        if image_type == 'mnist':
            test_images, test_labels = load_mnist(test_dir, 'validation')
        elif image_type == 'imageNet':
            test_images, test_labels = load_imageNet(graph_dir, 'validation')
        elif image_type == 'cifar10':
            test_images, test_labels = load_cifar10(graph_dir, 'validation')
        else:
            train_images, train_labels, test_images, test_labels = None, None, None, None
            self.logger.error('Not valid image type')
        # Generate Training data
        train_images, train_labels = self.generate_fake_images()
        if train_images is not None and len(train_images) > 0:
            self.train = DataSet(train_images, train_labels)
            self.logger.info('Train dataset shape: {:s} with min {:f}, max {:f}'
                             .format((self.train.images.shape,),
                                     self.train.images.min(), self.train.images.max()))
        if test_images is not None and len(test_images) > 0:
            self.test = DataSet(test_images, test_labels)
            self.logger.info('Test dataset shape: {:s} with min {:f}, max {:f}'
                             .format((self.test.images.shape,),
                                     self.test.images.min(), self.test.images.max()))
    def generate_fake_images(self):
        train_images, train_labels = None, None
        for edge_id, record_dir in np.column_stack((self.edge_ids, self.record_dir_sets)):
            self.logger.debug('~~~~~~~~~~~ For Edge %s ~~~~~~~~~~~~' % edge_id)
            self.logger.debug('Load from %s' % record_dir)
            # Reload the generator network
            sess = tf.Session()
            saver = tf.train.import_meta_graph(record_dir + '.meta')
            saver.restore(sess, record_dir)
            # Get essential input and output tensor
            graph = tf.get_default_graph()
            z = graph.get_tensor_by_name('input_' + str(edge_id) + '/z:0')
            self.logger.debug('z: %s' % (z,))
            gx = graph.get_tensor_by_name('gnet_' + str(edge_id) + "/gout:0")
            self.logger.debug('gout: %s' % (gx,))
            gl = graph.get_tensor_by_name('gnet_' + str(edge_id) + "/lout:0")
            self.logger.debug('lout: %s' % (gl,))
            # Generate images
            latent_dimensions = 10
            train_images_edge, train_labels_edge = \
                sess.run([gx, gl],
                         feed_dict={z: np.random.randn(self.num_images, latent_dimensions)})
            train_images_edge = np.clip(train_images_edge, 0., 1.) * 255. + 0.5
            train_classes_edge = np.argmax(train_labels_edge, axis=1)
            visualization_grey(train_images_edge, train_classes_edge, 'fake_' + edge_id)
            if train_images is None:
                train_images, train_labels = train_images_edge, train_classes_edge
            else:
                train_images, train_labels = np.concatenate([train_images, train_images_edge], axis=0), \
                                             np.concatenate([train_labels, train_classes_edge], axis=0),
            self.logger.debug("Generated data shape: %s" % (train_images.shape,))
            sess.close()
        return train_images, train_labels
    def update_images(self):
        train_images, train_labels = self.generate_fake_images()
        if train_images is not None and len(train_images) > 0:
            self.train = DataSet(train_images, train_labels)
            self.logger.info('Train dataset shape: {:s} with min {:f}, max {:f}'
                             .format((self.train.images.shape,),
                                     self.train.images.min(), self.train.images.max()))