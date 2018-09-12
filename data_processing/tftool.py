
import tensorflow as tf
from PIL import Image

import os, sys, pickle
import numpy as np
import glob

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('dataset_seed', 1234, "")

#

classes=['none','en','es','ja','fr','it','de','ko']

# ftrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
# writer= tf.python_io.TFRecordWriter(filepath+ftrecordfilename)




def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytesList_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floatList_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_images_and_labels(images, widths,heights,labels,languages,nlines,bbxs, filepath):
    num_examples = len(labels)
    assert len(images) == len(labels) == len(widths) == len(heights) == len(languages) == len(nlines) == len(bbxs),\
   "Images size %d does not match label size %d." %(len(images), num_examples)
    print('Writing', filepath, 'num_examples:',num_examples)
    writer = tf.python_io.TFRecordWriter(filepath)
    for index in range(num_examples):
        image_feature = _bytesList_feature(images[index])
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(widths[index]),
            'width': _int64_feature(heights[index]),
            'depth': _int64_feature(3),
            'label': _int64_feature(int(labels[index])),
            'language':_int64_feature(int(languages[index])),
            'nlines':_int64_feature(nlines[index]),
            'bbx': _floatList_feature(bbxs[index]),
            'image': image_feature}))
        writer.write(example.SerializeToString())
    writer.close()
#
#
# def read(filename_queue):
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(
#         serialized_example,
#         # Defaults are not specified since both keys are required.
#         features={
#             'image': tf.FixedLenFeature([3072], tf.float32),
#             'label': tf.FixedLenFeature([], tf.int64),
#         })
#
#     # Convert label from a scalar uint8 tensor to an int32 scalar.
#     image = features['image']
#     image = tf.reshape(image, [32, 32, 3])
#     label = tf.one_hot(tf.cast(features['label'], tf.int32), 10)
#     return image, label
#
#
# def generate_batch(
#         example,
#         min_queue_examples,
#         batch_size, shuffle):
#     """
#     Arg:
#         list of tensors.
#     """
#     num_preprocess_threads = 1
#
#     if shuffle:
#         ret = tf.train.shuffle_batch(
#             example,
#             batch_size=batch_size,
#             num_threads=num_preprocess_threads,
#             capacity=min_queue_examples + 3 * batch_size,
#             min_after_dequeue=min_queue_examples)
#     else:
#         ret = tf.train.batch(
#             example,
#             batch_size=batch_size,
#             num_threads=num_preprocess_threads,
#             allow_smaller_final_batch=True,
#             capacity=min_queue_examples + 3 * batch_size)
#
#     return ret


# def transform(image):
#     image = tf.reshape(image, [32, 32, 3])
#     if FLAGS.aug_trans or FLAGS.aug_flip:
#         print("augmentation")
#         if FLAGS.aug_trans:
#             image = tf.pad(image, [[2, 2], [2, 2], [0, 0]])
#             image = tf.random_crop(image, [32, 32, 3])
#         if FLAGS.aug_flip:
#             image = tf.image.random_flip_left_right(image)
#     return image

#
# def generate_filename_queue(filenames, data_dir, num_epochs=None):
#     print("filenames in queue:", filenames)
#     for i in range(len(filenames)):
#         filenames[i] = os.path.join(data_dir, filenames[i])
#     return tf.train.string_input_producer(filenames, num_epochs=num_epochs)



def prepare_dataset(lang,img_path,save_path):
    bestnum = 1000
    recordfilenum = 0
    all_image_paths =  glob.glob(img_path+'/*/'+'*.png')
    neg_image_paths =[ i for i in  all_image_paths if 'neg' in i]
    pos_image_paths = list(set(all_image_paths)-set(neg_image_paths))
    print "len(all_image_paths),len(neg_image_paths),len(pos_image_paths)",len(all_image_paths),len(neg_image_paths),len(pos_image_paths)



    rng = np.random.RandomState(FLAGS.dataset_seed)
    rand_idx = rng.permutation(len(neg_image_paths))


    neg_train_size = int(len(neg_image_paths)*0.8)

    rand_idx_train = rand_idx[:neg_train_size]
    rand_idx_test = rand_idx[neg_train_size:]
    neg_image_paths_train = [neg_image_paths[i] for i in rand_idx_train]
    neg_image_paths_test = [neg_image_paths[i] for i in rand_idx_test]
    print "len(neg_image_paths_train),len( neg_image_paths_test)", len(neg_image_paths_train),len( neg_image_paths_test)

    num =0
    neg_images =[]
    neg_widths =[]
    neg_heights =[]
    recordfilenum = 0
    for i,neg_image_path in enumerate(neg_image_paths_train):


        img = Image.open(neg_image_path, 'r')
        # img = img.resize((256,256),Image.ANTIALIAS)
        size = img.size
        # print(size[1], size[0])
        # print(size)
        # print(img.mode)
        img_raw = img.tobytes()
        neg_images += [img_raw]
        neg_widths += [size[0]]
        neg_heights += [size[1]]
        num = num + 1
        if num >= bestnum or i == len(neg_image_paths_train)-1:
            neg_labels = [0] * num
            languages = [0] * num
            nlines = [0] * num
            bbxs = [[] for _ in xrange(num)]


            ftrecordfilename = save_path + ("/traindata_neg_%d.tfrecords-%.3d" % (num,recordfilenum))
            convert_images_and_labels(neg_images, neg_widths, neg_heights, neg_labels, languages, nlines, bbxs,
                                      ftrecordfilename)
            recordfilenum = recordfilenum + 1
            num = 0
            neg_images = []
            neg_widths = []
            neg_heights = []

    num = 0
    recordfilenum = 0
    for i,neg_image_path_test in enumerate(neg_image_paths_test):

        img = Image.open(neg_image_path_test, 'r')
        # img = img.resize((256, 256), Image.ANTIALIAS)
        size = img.size
        # print(img.mode)
        img_raw = img.tobytes()
        neg_images += [img_raw]
        neg_widths += [size[0]]
        neg_heights += [size[1]]
        num = num + 1
        if num >= bestnum or i == len(neg_image_paths_test)-1 :
            neg_labels = [0] * num
            languages = [0] * num
            nlines = [0] * num
            bbxs = [[] for _ in xrange(num)]


            ftrecordfilename = save_path + ("/testdata_neg_%d.tfrecords-%.3d" % (num,recordfilenum))
            convert_images_and_labels(neg_images, neg_widths, neg_heights, neg_labels, languages, nlines, bbxs,
                                      ftrecordfilename)
            recordfilenum = recordfilenum + 1
            neg_images = []
            neg_widths = []
            neg_heights = []
            num = 0


    rng = np.random.RandomState(FLAGS.dataset_seed)
    rand_idx = rng.permutation(len(pos_image_paths))
    pos_train_size = int(len(pos_image_paths) * 0.8)

    rand_idx_train = rand_idx[:pos_train_size]
    rand_idx_test = rand_idx[pos_train_size:]
    pos_image_paths_train = [pos_image_paths[i] for i in rand_idx_train]
    pos_image_paths_test = [pos_image_paths[i] for i in rand_idx_test]
    num = 0
    pos_images = []
    pos_widths = []
    pos_heights = []
    nlines = []
    bbxs = []
    recordfilenum = 0
    for i,pos_image_path in enumerate(pos_image_paths_train):


        img = Image.open(pos_image_path, 'r')
        # img = img.resize((256, 256), Image.ANTIALIAS)
        size = img.size
        # print(size[1], size[0])
        # print(size)
        with open(pos_image_path[:-4]+'.pkl','rb') as f:
            bbx = pickle.load(f)
            # print "bbx, type(bbx)",bbx, type(bbx)

        f.close()
        assert len(bbx) >0, "no text in this frame!"
        nlines += [len(bbx)]
        bbxs.append(sum(bbx, []))
        # print(img.mode)
        img_raw = img.tobytes()
        pos_images += [img_raw]
        pos_widths += [size[0]]
        pos_heights += [size[1]]
        num = num + 1
        if num >= bestnum or i == len(pos_image_paths_train)-1 :
            pos_labels = [1] * num
            pos_languages = [classes.index(lang)] * num

            ftrecordfilename = save_path + ("/traindata_pos_%s_%d.tfrecords-%.3d" % (lang,num, recordfilenum))
            convert_images_and_labels(pos_images, pos_widths, pos_heights, pos_labels, pos_languages, nlines, bbxs,
                                     ftrecordfilename)
            recordfilenum = recordfilenum + 1
            num = 0
            pos_images = []
            pos_widths = []
            pos_heights = []
            nlines = []
            bbxs = []

    num = 0
    recordfilenum = 0
    for i,pos_image_path in enumerate(pos_image_paths_test):
        img = Image.open(pos_image_path, 'r')
        # img = img.resize((256, 256), Image.ANTIALIAS)
        size = img.size
        # print(size[1], size[0])
        # print(size)
        with open(pos_image_path[:-4] + '.pkl', 'rb') as f:
            bbx = pickle.load(f)
            # print "bbx, type(bbx)", bbx, type(bbx)

        f.close()
        assert len(bbx) > 0, "no text in this frame!"
        nlines += [len(bbx)]
        bbxs.append(sum(bbx, []))
        # print(img.mode)
        img_raw = img.tobytes()
        pos_images += [img_raw]
        pos_widths += [size[0]]
        pos_heights +=[size[1]]
        num = num + 1
        if num >= bestnum or num == len(pos_image_paths_test):
            pos_labels = [1] * num
            pos_languages = [classes.index(lang)] * num
            ftrecordfilename = save_path + ("/testdata_pos_%s_%d.tfrecords-%.3d" % (lang,num, recordfilenum))
            convert_images_and_labels(pos_images, pos_widths, pos_heights, pos_labels, pos_languages, nlines, bbxs,
                                     ftrecordfilename)
            recordfilenum = recordfilenum + 1
            num = 0
            pos_images = []
            pos_widths = []
            pos_heights = []
            nlines = []
            bbxs = []






#
# def inputs(batch_size=100,
#            train=True, validation=False,
#            shuffle=True, num_epochs=None):
#     if validation:
#         if train:
#             filenames = ['labeled_train_val.tfrecords']
#             num_examples = FLAGS.num_labeled_examples - FLAGS.num_valid_examples
#         else:
#             filenames = ['test_val.tfrecords']
#             num_examples = FLAGS.num_valid_examples
#     else:
#         if train:
#             filenames = ['labeled_train.tfrecords']
#             num_examples = FLAGS.num_labeled_examples
#         else:
#             filenames = ['test.tfrecords']
#             num_examples = NUM_EXAMPLES_TEST
#
#     filenames = [os.path.join('seed' + str(FLAGS.dataset_seed), filename) for filename in filenames]


if __name__ == '__main__':
    # for lang in (['en','de','es','fr','it']):
        lang = 'ja'
        print lang
        img_path = '../data/pngdataset/images/{}/'.format(lang)
        save_path =  '../data/tfdataset/ori_size/{}/'.format(lang)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        prepare_dataset(lang, img_path, save_path)