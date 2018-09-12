'''
    convert.py: convert CIFAR-10 dataset from binary batches to
                folders of individual images

'''

import argparse
import struct
import tensorflow as tf
import numpy as np
import os
import cv2
import cPickle

'populate class list'
classlist = []
classlist.append('airplane')
classlist.append('automobile')
classlist.append('bird')
classlist.append('cat')
classlist.append('deer')
classlist.append('dog')
classlist.append('frog')
classlist.append('horse')
classlist.append('ship')
classlist.append('truck')

def load_dfile(datafile):
    
    with open(datafile, 'rb') as f:
	dicttmp = cPickle.load(f)
    f.close()
	    
    dtmp = dicttmp['data']
    ltmp = dicttmp['labels']
    nimg = len(ltmp)
            
    per_images = np.zeros((nimg, 32, 32, 3))
    per_labels = np.zeros(nimg, dtype=np.uint8)
    for k in range(nimg):
	rtmp = dtmp[k,0:32*32]
	gtmp = dtmp[k,32*32:2*32*32]
	btmp = dtmp[k,2*32*32:]
	per_images[k,:,:,0]=np.reshape(rtmp, (32,32))    
	per_images[k,:,:,1]=np.reshape(gtmp, (32,32))    
	per_images[k,:,:,2]=np.reshape(btmp, (32,32))
	per_labels[k]=ltmp[k] 
    
    return per_images, per_labels

def load_dataset(basedir, tag): 
    d = None
    l = None

    if tag=='train': 
	for i in range(5): 
	    datafile = os.path.join(basedir, 'data_batch_%d' % (i+1))
	    images, labels = load_dfile(datafile)
	    if d is None:
		d = images
		l = labels 
	    else: 
		d = np.concatenate([d, images], axis=0)
		l = np.concatenate([l, labels], axis=0)
		
    elif tag=='test': 
	datafile = os.path.join(basedir, 'test_batch')
	d, l= load_dfile(datafile)
    else:
	print 'wrong tag: ', tag

    # d = d/255. - 0.5
    return d, l

def save_dataset(basedir, tag, d, l): 
    '''
	save images to target directories
    '''  
    dstdir = os.path.join(basedir, tag)
    nimg = len(l)
    for i in range(nimg): 
	idx = l[i]
	fname = '%s/%06d.bmp' % (classlist[idx], i)
	imgfile = os.path.join(dstdir, fname) 
	print 'save to image file', imgfile
 	im = d[i]
        cv2.imwrite(imgfile, im)        

# for converting to tf records
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def save_tfrecord(basedir, tag, images, labels):
    '''
	save images to target tfrecords
    '''

    assert images.shape[0] == labels.shape[0]
    num = images.shape[0]
    
    'convert 32x32x3 => vector | [0,255] => [-0.5,0.5]'
    images = images.reshape((num, -1))/255.-0.5

    tffile = os.path.join(basedir, '%s-new.tfrecords' % tag) 

    print 'start writing to tfrecord: ', tffile
    writer = tf.python_io.TFRecordWriter(tffile)
    for i in range(num):
        image = images[i].tolist()
        image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(32),
            'width': _int64_feature(32),
            'depth': _int64_feature(3),
            'label': _int64_feature(int(labels[i])),
            'image': image_feature}))
        writer.write(example.SerializeToString())
    writer.close()

    print 'completed writing to tfrecord: ', tffile


# --------- start of main procedures -----------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tag',    help='directory of dataset (train | test)',  default ='test')
parser.add_argument('--srcdir', help='source directory of dataset',
				default='/home/xiaoqzhu/dataset/cifar-10/cifar-10-batches-py')
parser.add_argument('--dstdir', help='destination directory of dataset', 
				default='/home/xiaoqzhu/dataset/cifar-10/cifar-10-imgs')
parser.add_argument('--tfdir', help='target directory of dataset in tfrecord', 
				default='/home/xiaoqzhu/dataset/cifar-10/cifar-10-tfrec')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args


'load data'
d, l = load_dataset(args.srcdir, args.tag)
args.nclass = max(l)+1
print 'd.shape: ', d.shape, ' l.shape: ', l.shape

'save images to target folders'
# save_dataset(args.dstdir, args.tag, d, l)

'save images to target tf records'
save_tfrecord(args.tfdir, args.tag, d, l)
