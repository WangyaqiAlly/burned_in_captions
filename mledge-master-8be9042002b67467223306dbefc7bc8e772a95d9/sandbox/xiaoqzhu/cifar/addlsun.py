'''
    addlsun.py:  add person images from LSUN (in h5, already resized to 32x32)
    as a new class for CIFAR-10
'''

import argparse
import struct
import tensorflow as tf
import numpy as np
import os
import cv2
import cPickle
import random
import h5py

def convert(args):
    
    h5 = h5py.File(args.src, 'r')
    lods = sorted([value for key, value in h5.iteritems() if key.startswith('data')], key=lambda lod: -lod.shape[3])

    dstdir = os.path.join(args.dstdir, args.tag)
    dstdir = os.path.join(dstdir, 'person')
    
    if args.tag=='train':     
	nmin = 0
	nmax = 15000000
        nimg = 5000
    else:
	nmin = 16000000 
	nmax = 18000000
	nimg = 1000

    idxlist = random.sample(xrange(nmin, nmax), nimg)
    print 'converting %d images' % len(idxlist)

    # indices = range(lods[0].shape[0])
    # indices = indices[start : stop : step]

    for idx in idxlist:
	bmpfile = '%06d.bmp' % idx
	dstfile = os.path.join(dstdir, bmpfile)
        print 'saving to ', dstfile
	# print '%d / %d' % (idx, lods[0].shape[0])
        img = lods[0][idx]
        img = img.transpose(1, 2, 0) # CHW => HWC
        img = img[:, :, ::-1] # RGB => BGR
	cv2.imwrite(dstfile, img)	
        # cv2.imwrite()

    h5.close()
    
def display(h5_filename, start=0, stop=10, step=None):

    print 'Displaying images from %s' % h5_filename
    h5 = h5py.File(h5_filename, 'r')
    lods = sorted([value for key, value in h5.iteritems() if key.startswith('data')], key=lambda lod: -lod.shape[3])
    indices = range(lods[0].shape[0])
    indices = indices[start : stop : step]

    # import cv2 # pip install opencv-python
    #window_name = 'h5tool'
    #cv2.namedWindow(window_name)
    #print 'Press SPACE or ENTER to advance, ESC to exit.'

    for idx in indices:
        print '%d / %d' % (idx, lods[0].shape[0])
        img = lods[0][idx]
        img = img.transpose(1, 2, 0) # CHW => HWC
        img = img[:, :, ::-1] # RGB => BGR
        cv2.imwrite('./testimg/test_{}.jpg'.format(idx),img)
        #cv2.imshow(window_name, img)
        #c = cv2.waitKey()
        #if c == 27:
        #    break

    h5.close()
    print '%-40s\r' % ''
    print 'Done.'

# --------- start of main procedures -----------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tag', help='directory of dataset',  default ='test')
parser.add_argument('--src', help='directory of dataset',
				 default='/home2/LSUN_yaqi/lsun-32x32-h5/topten/lsun-person-32x32.h5')
parser.add_argument('--dstdir', help='directory of dataset', 
				 default='/home/xiaoqzhu/dataset/cifar-10/cifar-10-imgs')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()

print args

# display(args.src)
convert(args)

