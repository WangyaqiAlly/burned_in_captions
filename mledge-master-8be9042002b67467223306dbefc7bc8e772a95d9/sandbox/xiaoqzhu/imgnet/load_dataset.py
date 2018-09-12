'''
    load_dataset:  
    load numpy arrays of training/validation datasets
    directly from pickle files

'''

import argparse
import struct
import tensorflow as tf
import numpy as np
import os
import cv2
import cPickle

classlist = []
classlist.append('invertebrate')
classlist.append('bird')
classlist.append('vehicle')
classlist.append('dog')
classlist.append('clothing')

def rshuffle(d, l): 

    'randomly shuffle dataset'
    ind = np.arange(d.shape[0])
    np.random.shuffle(ind)
    d = d[ind]
    l = l[ind]
    return d, l

# --------- start of main procedures -----------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nt', help='number of training subsets', default=4, type=int)
parser.add_argument('--h', help='image height', default=32, type=int)
parser.add_argument('--w', help='image width', default=32, type=int)
parser.add_argument('--d', help='depth', default=3, type=int)
parser.add_argument('--datadir', help='directory of dataset', 
				 default='/home/xiaoqzhu/dataset/imagenet-plus-5-class-cubic/pfiles')
args = parser.parse_args()
print args

'load training data'
d = np.array([]).reshape([0, args.h, args.w, args.d])
l = np.array([])
for k in range(args.nt): 
    dfilename = 'train-images-%02d.pickle' % k
    lfilename = 'train-labels-%02d.pickle' % k
    dfile = os.path.join(args.datadir, dfilename)
    lfile = os.path.join(args.datadir, lfilename)

    with open(dfile, 'rb') as f: 
        dtmp = cPickle.load(f)
    f.close()

    with open(lfile, 'rb') as f: 
        ltmp = cPickle.load(f)
    f.close()

    # Concatenate along axis 0 by default
    d = np.concatenate((d, dtmp))
    l = np.concatenate((l, ltmp))

'load validation data'
dtfile = os.path.join(args.datadir, 'test-images.pickle')
ltfile = os.path.join(args.datadir, 'test-labels.pickle')
with open(dtfile, 'rb') as f:
    dt = cPickle.load(f)
f.close()
with open(ltfile, 'rb') as f:
    lt = cPickle.load(f)
f.close()

print 'training: d.shape: ', d.shape, ' l.shape: ', l.shape
print 'validation: ',  dt.shape, ' dt.shape: ',  lt.shape

'random shuffle training & validation data once'
d, l = rshuffle(d, l)
dt, lt = rshuffle(dt, lt)
	

