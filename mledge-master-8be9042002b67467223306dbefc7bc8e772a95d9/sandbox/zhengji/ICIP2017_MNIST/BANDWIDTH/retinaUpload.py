import h5py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

STANDARD_MODE = 1
OPTIMIZED_ABCD_MODE = 2
OPTIMIZED_WR_MODE = 3

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
        if (int(len(correct_images) * 0.5) == len(correct_images)):
            threshold = -1
        else:
            threshold = temp[int(len(correct_images) * 0.5)] * -1
        under_cross_entropy = np.ix_(cross_entropy <= threshold)[0]
        A_images = correct_images[under_cross_entropy,:]
        A_labels = correct_labels[under_cross_entropy]
        A_metric = cross_entropy[under_cross_entropy].copy()
        over_cross_entropy = np.ix_(cross_entropy > threshold)[0]
        B_images = correct_images[over_cross_entropy,:]
        B_labels = correct_labels[over_cross_entropy]
        B_metric = cross_entropy[over_cross_entropy].copy()
        # C & D
        cross_entropy = network_obj.sess.run(network_obj.cross_entropy_individual, feed_dict={
            network_obj.x: wrong_images,
            network_obj.y_: wrong_labels,
            network_obj.keep_prob: 1.0
            })
        temp = cross_entropy.copy() * -1
        temp.sort()
        if (int(len(wrong_images) * 0.5) == len(wrong_images)):
            threshold = -1
        else:
            threshold = temp[int(len(wrong_images) * 0.5)] * -1
        over_cross_entropy = np.ix_(cross_entropy > threshold)[0]
        C_images = wrong_images[over_cross_entropy,:]
        C_labels = wrong_labels[over_cross_entropy]
        C_metric = cross_entropy[over_cross_entropy].copy()

        under_cross_entropy = np.ix_(cross_entropy <= threshold)[0]
        D_images = wrong_images[under_cross_entropy,:]
        D_labels = wrong_labels[under_cross_entropy]
        D_metric = cross_entropy[under_cross_entropy].copy()

        amount = rate * len(train_images)
        print 'amount: ', amount
        amount_A = int(round(amount * len(A_images) / len(train_images)))
        amount_B = int(round(amount * len(B_images) / len(train_images)))
        amount_C = int(round(amount * len(C_images) / len(train_images)))
        amount_D = int(round(amount * len(D_images) / len(train_images)))
        print 'amount_ABCD: ', amount_A + amount_B + amount_C + amount_D
        print 'train_images: ', len(train_images)
        print 'a: %d | b: %d | c: %d | d: %d ', (len(A_images) , len(B_images) , len(C_images) , len(D_images))

        if os.path.exists(os.path.join(dest_path, 'queue.hdf')):
            print 'reading data from %s' % os.path.join(dest_path, 'queue.hdf')
            f = h5py.File(os.path.join(dest_path, 'queue.hdf'), 'r')
            A_images = np.concatenate((A_images, np.array(f.get('A_images'))), axis = 0)
            A_labels = np.concatenate((A_labels, np.array(f.get('A_labels'))), axis = 0)
            A_metric = np.concatenate((A_metric, np.array(f.get('A_metric'))), axis = 0)
            B_images = np.concatenate((B_images, np.array(f.get('B_images'))), axis = 0)
            B_labels = np.concatenate((B_labels, np.array(f.get('B_labels'))), axis = 0)
            B_metric = np.concatenate((B_metric, np.array(f.get('B_metric'))), axis = 0)
            C_images = np.concatenate((C_images, np.array(f.get('C_images'))), axis = 0)
            C_labels = np.concatenate((C_labels, np.array(f.get('C_labels'))), axis = 0)
            C_metric = np.concatenate((C_metric, np.array(f.get('C_metric'))), axis = 0)
            D_images = np.concatenate((D_images, np.array(f.get('D_images'))), axis = 0)
            D_labels = np.concatenate((D_labels, np.array(f.get('D_labels'))), axis = 0)
            D_metric = np.concatenate((D_metric, np.array(f.get('D_metric'))), axis = 0)
            f.close()

        temp = [[m, i ,l] for m, i, l in zip(A_metric, A_images, A_labels)]
        temp = sorted(temp, key=lambda item: item[0])
        A_metric = np.array([item[0] for item in temp])
        A_images = np.array([item[1] for item in temp])
        A_labels = np.array([item[2] for item in temp])

        temp = [[m, i ,l] for m, i, l in zip(B_metric, B_images, B_labels)]
        temp = sorted(temp, key=lambda item: item[0], reverse=True)
        B_metric = np.array([item[0] for item in temp])
        B_images = np.array([item[1] for item in temp])
        B_labels = np.array([item[2] for item in temp])

        temp = [[m, i ,l] for m, i, l in zip(C_metric, C_images, C_labels)]
        temp = sorted(temp, key=lambda item: item[0], reverse=True)
        C_metric = np.array([item[0] for item in temp])
        C_images = np.array([item[1] for item in temp])
        C_labels = np.array([item[2] for item in temp])

        temp = [[m, i ,l] for m, i, l in zip(D_metric, D_images, D_labels)]
        temp = sorted(temp, key=lambda item: item[0])
        D_metric = np.array([item[0] for item in temp])
        D_images = np.array([item[1] for item in temp])
        D_labels = np.array([item[2] for item in temp])

        if mode==STANDARD_MODE:
            if os.path.exists(os.path.join(dest_path, 'queue_standard.hdf')):
                f = h5py.File(os.path.join(dest_path, 'queue_standard.hdf'), 'r')
                train_images = np.concatenate((np.array(f.get('images')), train_images), axis = 0)
                train_labels = np.concatenate((np.array(f.get('labels')), train_labels), axis = 0)
                f.close()
            temp_images = train_images[int(amount):, :]
            temp_labels = train_labels[int(amount):]
            print 'writing queue standard data into %s | %d' % (os.path.join(dest_path, 'queue_standard.hdf'), len(temp_images))
            f = h5py.File(os.path.join(dest_path, 'queue_standard.hdf'), 'w')
            f.create_dataset('images', data = temp_images, dtype = 'f')
            f.create_dataset('labels', data = temp_labels, dtype = 'i')
            f.close()
            train_images = train_images[:int(amount), :]
            train_labels = train_labels[:int(amount)]
        elif mode==OPTIMIZED_ABCD_MODE:
            left = int(amount)
            train_images = None
            train_labels = None
            send_num = min(left, len(D_images))
            if send_num > 0:
                if train_images == None:
                    train_images = D_images[:send_num, :]
                    train_labels = D_labels[:send_num]
                else:
                    train_images = np.concatenate((train_images, D_images[:send_num, :]), axis=0)
                    train_labels = np.concatenate((train_labels, D_labels[:send_num]), axis=0)
                D_images = D_images[send_num:, :]
                D_labels = D_labels[send_num:]
                D_metric = D_metric[send_num:]
                left -= send_num
            send_num = min(left, len(C_images))
            if send_num > 0:
                if train_images == None:
                    train_images = C_images[:send_num, :]
                    train_labels = C_labels[:send_num]
                else:
                    train_images = np.concatenate((train_images, C_images[:send_num, :]), axis=0)
                    train_labels = np.concatenate((train_labels, C_labels[:send_num]), axis=0)
                C_images = C_images[send_num:, :]
                C_labels = C_labels[send_num:]
                C_metric = C_metric[send_num:]
                left -= send_num
            send_num = min(left, len(B_images))
            if send_num > 0:
                if train_images == None:
                    train_images = B_images[:send_num, :]
                    train_labels = B_labels[:send_num]
                else:
                    train_images = np.concatenate((train_images, B_images[:send_num, :]), axis=0)
                    train_labels = np.concatenate((train_labels, B_labels[:send_num]), axis=0)
                B_images = B_images[send_num:, :]
                B_labels = B_labels[send_num:]
                B_metric = B_metric[send_num:]
                left -= send_num
            send_num = min(left, len(A_images))
            if send_num > 0:
                if train_images == None:
                    train_images = A_images[:send_num, :]
                    train_labels = A_labels[:send_num]
                else:
                    train_images = np.concatenate((train_images, A_images[:send_num, :]), axis=0)
                    train_labels = np.concatenate((train_labels, A_labels[:send_num]), axis=0)
                A_images = A_images[send_num:, :]
                A_labels = A_labels[send_num:]
                A_metric = A_metric[send_num:]
                left -= send_num
                
        elif mode==OPTIMIZED_WR_MODE:
            if os.path.exists(os.path.join(dest_path, 'queue_wr.hdf')):
                f = h5py.File(os.path.join(dest_path, 'queue_wr.hdf'), 'r')
                correct_images = np.concatenate((np.array(f.get('correct_images')), correct_images), axis = 0)
                correct_labels = np.concatenate((np.array(f.get('correct_labels')), correct_labels), axis = 0)
                wrong_images = np.concatenate((np.array(f.get('wrong_images')), wrong_images), axis = 0)
                wrong_labels = np.concatenate((np.array(f.get('wrong_labels')), wrong_labels), axis = 0)
                f.close()
            left_num = int(amount)
            train_images = None
            train_labels = None
            send_num = min(left_num, len(wrong_images))
            if send_num > 0:
                if train_images == None:
                    train_images = wrong_images[:send_num, :]
                    train_labels = wrong_labels[:send_num]
                else:
                    train_images = np.concatenate((train_images, wrong_images[:send_num, :]), axis=0)
                    train_labels = np.concatenate((train_labels, wrong_labels[:send_num]), axis=0)
                wrong_images = wrong_images[send_num:, :]
                wrong_labels = wrong_labels[send_num:]
                left_num -= send_num

            send_num = min(left_num, len(correct_images))
            if (send_num > 0):
                if (train_images == None):
                    train_images = correct_images[:send_num, :]
                    train_labels = correct_labels[:send_num]
                else:
                    train_images = np.concatenate((train_images, correct_images[:send_num, :]), axis=0)
                    train_labels = np.concatenate((train_labels, correct_labels[:send_num]), axis=0)
                correct_images = correct_images[send_num:, :]
                correct_labels = correct_labels[send_num:]
                left_num -= send_num
            print "left : %d" % (left_num)

            print 'writing queue wr data into %s | %d %d' % (os.path.join(dest_path, 'queue_wr.hdf'), len(correct_images), len(wrong_images))
            f = h5py.File(os.path.join(dest_path, 'queue_wr.hdf'), 'w')
            f.create_dataset('correct_images', data = correct_images, dtype = 'f')
            f.create_dataset('correct_labels', data = correct_labels, dtype = 'i')
            f.create_dataset('wrong_images', data = wrong_images, dtype = 'f')
            f.create_dataset('wrong_labels', data = wrong_labels, dtype = 'i')
            f.close()

        print 'writing queue data into %s | %d %d %d %d' % (os.path.join(dest_path, 'queue.hdf'), len(A_images), len(B_images), len(C_images), len(D_images))
        f = h5py.File(os.path.join(dest_path, 'queue.hdf'), 'w')
        f.create_dataset('A_images', data = A_images, dtype = 'f')
        f.create_dataset('A_labels', data = A_labels, dtype = 'i')
        f.create_dataset('A_metric', data = A_metric, dtype = 'f')
        f.create_dataset('B_images', data = B_images, dtype = 'f')
        f.create_dataset('B_labels', data = B_labels, dtype = 'i')
        f.create_dataset('B_metric', data = B_metric, dtype = 'f')
        f.create_dataset('C_images', data = C_images, dtype = 'f')
        f.create_dataset('C_labels', data = C_labels, dtype = 'i')
        f.create_dataset('C_metric', data = C_metric, dtype = 'f')
        f.create_dataset('D_images', data = D_images, dtype = 'f')
        f.create_dataset('D_labels', data = D_labels, dtype = 'i')
        f.create_dataset('D_metric', data = D_metric, dtype = 'f')
        f.close()

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


    


