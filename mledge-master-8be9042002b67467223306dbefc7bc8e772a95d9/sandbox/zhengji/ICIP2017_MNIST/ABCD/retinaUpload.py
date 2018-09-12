import h5py
import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
STANDARD_MODE= 1
A_MODE = 2
B_MODE = 3
C_MODE = 4
D_MODE = 5

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
        # A & B
        cross_entropy = network_obj.sess.run(network_obj.cross_entropy_individual, feed_dict={
            network_obj.x: correct_images,
            network_obj.y_: correct_labels,
            network_obj.keep_prob: 1.0
            })
        temp = cross_entropy.copy() * -1
        temp.sort()
        if (int(len(correct_images) * rate) == len(correct_images)):
            threshold = -1
        else:
            threshold = temp[int(len(correct_images) * rate)] * -1
        max_num = int(len(correct_images) * rate)
        under_cross_entropy = np.ix_(cross_entropy <= threshold)[0][:max_num]
        A_images = correct_images[under_cross_entropy,:]
        A_labels = correct_labels[under_cross_entropy]
        over_cross_entropy = np.ix_(cross_entropy > threshold)[0][:max_num]
        B_images = correct_images[over_cross_entropy,:]
        B_labels = correct_labels[over_cross_entropy]
        # C & D
        cross_entropy = network_obj.sess.run(network_obj.cross_entropy_individual, feed_dict={
            network_obj.x: wrong_images,
            network_obj.y_: wrong_labels,
            network_obj.keep_prob: 1.0
            })
        temp = cross_entropy.copy() * -1
        temp.sort()
        if (int(len(wrong_images) * rate) == len(wrong_images)):
            threshold = -1
        else:
            threshold = temp[int(len(wrong_images) * rate)] * -1
        max_num = int(len(wrong_images) * rate)
        over_cross_entropy = np.ix_(cross_entropy > threshold)[0][:max_num]
        C_images = wrong_images[over_cross_entropy,:]
        C_labels = wrong_labels[over_cross_entropy]
        #print cross_entropy[over_cross_entropy]

        under_cross_entropy = np.ix_(cross_entropy <= threshold)[0][:max_num]
        D_images = wrong_images[under_cross_entropy,:]
        D_labels = wrong_labels[under_cross_entropy]
        #print cross_entropy[under_cross_entropy]

        if mode==A_MODE:
            train_images = A_images
            train_labels = A_labels
        elif mode==B_MODE:
            train_images = B_images
            train_labels = B_labels
        elif mode==C_MODE:
            train_images = C_images
            train_labels = C_labels
        elif mode==D_MODE:
            train_images = D_images
            train_labels = D_labels
        elif mode==STANDARD_MODE:
            pass

        correctnum = len(correct)
        wrongnum = len(wrong)
        forwardnum = len(train_images)

        # check or produce new subdir in destination
        if os.path.exists(dest_path):
            temp_images = train_images.copy()
            temp_labels = train_labels.copy()
            if os.path.exists(os.path.join(dest_path, 'received.hdf')):
                print 'reading data from %s' % os.path.join(dest_path, 'received.hdf')
                f = h5py.File(os.path.join(dest_path, 'received.hdf'), 'r')
                temp_images = np.concatenate((temp_images, np.array(f.get('images'))), axis = 0)
                temp_labels = np.concatenate((temp_labels, np.array(f.get('labels'))), axis = 0)
                f.close()
            print 'writing retesting data into %s | %d' % (dest_path, len(temp_images))
            f = h5py.File(os.path.join(dest_path, 'received.hdf'), 'w')
            f.create_dataset('images', data = temp_images, dtype = 'f')
            f.create_dataset('labels', data = temp_labels, dtype = 'i')
            f.close()

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


    


