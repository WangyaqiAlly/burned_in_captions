import h5py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

RANDOM_MODE = 1
ENTROPY_MODE = 2
CERTAINTY_MODE = 3
SRN1_MODE = 4
SRN2_MODE = 5
CROSS_ENTROPY_MODE=6


class Uploader(object):
    def __init__(self, source, dest, subdir_source='default', subdir_dest='default'):
        self.source = source
        self.dest = dest
        self.subdir_source = subdir_source
        self.subdir_dest = subdir_dest
        assert os.path.exists(source), ("path %s is not correct" % (source))
        assert os.path.exists(dest), ("path %s is not correct" % (dest))

    def write_hdf(self, images, labels, dest_path):
        print 'writing data into %s | %d' % (dest_path, len(images))
        f = h5py.File(os.path.join(dest_path, 'train.hdf'), 'w')
        f.create_dataset('images', data = images, dtype = 'f')
        f.create_dataset('labels', data = labels, dtype = 'i')
        f.close()


    def create_record(self, images, labels, path):
        writer = tf.python_io.TFRecordWriter(os.path.join(path, "train.tfrecords"))
        for image, label in zip(images, labels):
            label = int(label)
            img_raw = image.astype('uint8').tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
        writer.close()


    def filt_and_transfer(self, filename, network_obj, rate=1.0, mode=0):
        source_path = os.path.join(self.source, self.subdir_source)
        dest_path   = os.path.join(self.dest, self.subdir_dest)
        # get data from source node
        print 'reading data from %s' % source_path
        f = h5py.File(os.path.join(source_path, filename), 'r')
        train_images = np.array(f.get('images'))
        train_labels = np.array(f.get('labels'))
        f.close()

        perm = np.arange(len(train_images))
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        #filt part
        correct_prediction = network_obj.sess.run(network_obj.correct_prediction, feed_dict={
            network_obj.x: train_images, 
            network_obj.y_: train_labels,
            network_obj.keep_prob: 1.0
            })
        correct_prediction = np.array(correct_prediction)
        correct = np.ix_(correct_prediction == True)[0]
        wrong   = np.ix_(correct_prediction == False)[0]
        correct_images = train_images[correct,:]
        correct_labels = train_labels[correct]
        wrong_images   = train_images[wrong,:]
        wrong_labels   = train_labels[wrong]

        amount = int(len(train_images) * rate) - len(wrong_images)

        if mode==ENTROPY_MODE:
            softmax = network_obj.sess.run(network_obj.softmax, feed_dict={
                network_obj.x: correct_images,
                network_obj.y_: correct_labels,
                network_obj.keep_prob: 1.0
                })
            softmax_log = np.log(softmax)
            entropy = np.sum(softmax * softmax_log * -1, axis = 1)
            temp = [[i, item] for i, item in enumerate(entropy)]
            temp = sorted(temp, key=lambda item: item[1], reverse=True)
            over_entropy = np.array([temp[i][0] for i in range(amount)])
            over_images = correct_images[over_entropy,:]
            over_labels = correct_labels[over_entropy]
            train_images = np.concatenate((wrong_images, over_images), axis=0)
            train_labels = np.concatenate((wrong_labels, over_labels), axis=0)
        elif mode==RANDOM_MODE:
            perm = np.arange(len(correct_images))
            np.random.shuffle(perm)
            correct_images = correct_images[perm]
            correct_labels = correct_labels[perm]
            select_num = amount
            train_images = np.concatenate((wrong_images, correct_images[:select_num]), axis=0)
            train_labels = np.concatenate((wrong_labels, correct_labels[:select_num]), axis=0)
        elif mode==CERTAINTY_MODE:
            softmax = network_obj.sess.run(network_obj.softmax, feed_dict={
                network_obj.x: correct_images,
                network_obj.y_: correct_labels,
                network_obj.keep_prob: 1.0
                })
            certainty = np.sum(softmax.astype('float64') ** 2, axis = 1)
            temp = [[i, item] for i, item in enumerate(certainty)]
            temp = sorted(temp, key=lambda item: item[1], reverse=False)
            over_certainty = np.array([temp[i][0] for i in range(amount)])
            over_images = correct_images[over_certainty,:]
            over_labels = correct_labels[over_certainty]
            train_images = np.concatenate((wrong_images, over_images), axis=0)
            train_labels = np.concatenate((wrong_labels, over_labels), axis=0)
        elif mode==SRN1_MODE:
            softmax = network_obj.sess.run(network_obj.softmax, feed_dict={
                network_obj.x: correct_images,
                network_obj.y_: correct_labels,
                network_obj.keep_prob: 1.0
                })
            snr1 = np.array([softmax[i, idx] for i, idx in zip(range(len(correct_images)),np.argmax(softmax, axis=1))])
            temp = [[i, item] for i, item in enumerate(snr1)]
            temp = sorted(temp, key=lambda item: item[1], reverse=False)
            over_snr = np.array([temp[i][0] for i in range(amount)])
            over_images = correct_images[over_snr,:]
            over_labels = correct_labels[over_snr]
            train_images = np.concatenate((wrong_images, over_images), axis=0)
            train_labels = np.concatenate((wrong_labels, over_labels), axis=0)
        elif mode==SRN2_MODE:
            softmax = network_obj.sess.run(network_obj.softmax, feed_dict={
                network_obj.x: correct_images,
                network_obj.y_: correct_labels,
                network_obj.keep_prob: 1.0
                })
            snr1 = np.array([softmax[i, idx] for i, idx in zip(range(len(correct_images)),np.argmax(softmax, axis=1))])
            snr2 = snr1 / (np.sum(softmax, axis=1) - snr1)
            temp = [[i, item] for i, item in enumerate(snr2)]
            temp = sorted(temp, key=lambda item: item[1], reverse=False)
            over_snr = np.array([temp[i][0] for i in range(amount)])
            over_images = correct_images[over_snr,:]
            over_labels = correct_labels[over_snr]
            train_images = np.concatenate((wrong_images, over_images), axis=0)
            train_labels = np.concatenate((wrong_labels, over_labels), axis=0)
        elif mode==CROSS_ENTROPY_MODE:
            cross_entropy = network_obj.sess.run(network_obj.cross_entropy_individual, feed_dict={
                network_obj.x: correct_images,
                network_obj.y_: correct_labels,
                network_obj.keep_prob: 1.0
                })
            temp = [[i, item] for i, item in enumerate(cross_entropy)]
            temp = sorted(temp, key=lambda item: item[1], reverse=True)
            over_cross_entropy= np.array([temp[i][0] for i in range(amount)])
            over_images = correct_images[over_cross_entropy,:]
            over_labels = correct_labels[over_cross_entropy]
            train_images = np.concatenate((wrong_images, over_images), axis=0)
            train_labels = np.concatenate((wrong_labels, over_labels), axis=0)

        correctnum = len(correct)
        wrongnum = len(wrong)
        forwardnum = len(train_images)

        # check or produce new subdir in destination
        if os.path.exists(dest_path):
            print 'reading data from %s' % dest_path
            f = h5py.File(os.path.join(dest_path, 'train.hdf'), 'r')
            train_images = np.concatenate((train_images, np.array(f.get('images'))), axis = 0)
            train_labels = np.concatenate((train_labels, np.array(f.get('labels'))), axis = 0)
            f.close()
        else:
            os.mkdir(dest_path)

        self.write_hdf(train_images, train_labels, dest_path)
        return wrongnum, correctnum, forwardnum

    def transfer(self, filename):
        source_path = os.path.join(self.source, self.subdir_source)
        dest_path   = os.path.join(self.dest, self.subdir_dest)
        # get data from source node
        print 'reading data from %s' % source_path
        f = h5py.File(os.path.join(source_path, filename), 'r')
        train_images = np.array(f.get('images'))
        train_labels = np.array(f.get('labels'))
        f.close()

        perm = np.arange(len(train_images))
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        # check or produce new subdir in destination
        if os.path.exists(dest_path):
            print 'reading data from %s' % dest_path
            f = h5py.File(os.path.join(dest_path, 'train.hdf'), 'r')
            train_images = np.concatenate((train_images, np.array(f.get('images'))), axis = 0)
            train_labels = np.concatenate((train_labels, np.array(f.get('labels'))), axis = 0)
            f.close()
        else:
            os.mkdir(dest_path)
        
        # write the train set 
        self.write_hdf(train_images, train_labels, dest_path)
        
        print 'finished.'


    


