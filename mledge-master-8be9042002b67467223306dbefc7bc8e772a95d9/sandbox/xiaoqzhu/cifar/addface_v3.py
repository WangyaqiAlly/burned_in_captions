'''
    addface.py:  add face images from LFW (after re-sizing) as a new class for CIFAR-10
'''

import argparse
import struct
import tensorflow as tf
import numpy as np
import os
import cv2
import cPickle
import random

def convert(args):

    if args.tag=='train':     
	nmin = 0
	nmax = 10000
        nimg = 5000
    else:
	nmin = 10001 
	nmax = 13233
	nimg = 1000

    idxlist = random.sample(xrange(nmin, nmax), nimg)
    print 'converting %d images' % len(idxlist)

    dstdir = os.path.join(args.dstdir, args.tag)
    dstdir = os.path.join(dstdir, 'face')
    imglist = os.listdir(args.srcdir)

    for i in idxlist:
	jpgfile = imglist[i] 
	bmpfile = jpgfile[:-3]+'bmp'
	srcfile = os.path.join(args.srcdir, jpgfile)
	dstfile = os.path.join(dstdir, bmpfile)
	print '%s -> %s' % (srcfile, dstfile)
 	
	im = cv2.imread(srcfile)
	print 'updated size: ', im.shape
	im2 = cv2.resize(im, (32, 32), interpolation = cv2.INTER_CUBIC)	
	cv2.imwrite(dstfile, im2)	

# --------- start of main procedures -----------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tag', help='directory of dataset',  default ='test')
parser.add_argument('--srcdir', help='directory of dataset',
				 default='/home/xiaoqzhu/dataset/lfw/lfw/face')
parser.add_argument('--dstdir', help='directory of dataset', 
				 default='/home/xiaoqzhu/dataset/cifar-10/cifar-10-imgs')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

convert(args)

