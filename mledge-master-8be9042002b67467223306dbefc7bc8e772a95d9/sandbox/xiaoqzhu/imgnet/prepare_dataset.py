'''
    parepare_dataset:  
    convert image folders to pickle files containing
    numpy arrays of training/validation datasets

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

def load_dataset(basedir, classlist, args): 

    nimglist = []
    for i, cname in enumerate(classlist): 
 	classdir = os.path.join(basedir, '%02d-%s' % (i, cname))
 	imglist = os.listdir(classdir)
	nimglist.append(len(imglist))

    print 'images per class: ', nimglist
 
    nimg = sum(nimglist)
    d = np.zeros((nimg,args.m, args.m, 3))
    l = np.zeros(nimg)
    idx = 0
    for i, cname in enumerate(classlist): 
 	classdir = os.path.join(basedir, '%02d-%s' % (i, cname))
 	imglist = os.listdir(classdir)
	for imgfile in imglist:
	    bmpfile = os.path.join(classdir, imgfile) 	    
	    img = cv2.imread(bmpfile)
	    print bmpfile, ' | ', img.shape
 	    
	    d[idx,:,:,:]=img
	    l[idx]=i
	    idx = idx + 1

    return d, l

def rshuffle(d, l): 

    'randomly shuffle dataset'
    ind = np.arange(d.shape[0])
    np.random.shuffle(ind)
    d = d[ind]
    l = l[ind]
    return d, l

# --------- start of main procedures -----------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='image size', default=32, type=int)
parser.add_argument('--datadir', help='directory of dataset', 
				 default='/home/xiaoqzhu/dataset/imagenet-plus-5-class-cubic')
args = parser.parse_args()
print args

args.nclass = len(classlist)

'load data'
traindir = os.path.join(args.datadir, 'train')
validdir = os.path.join(args.datadir, 'validation')

train_data, train_labels = load_dataset(traindir, classlist, args)
valid_data, valid_labels = load_dataset(validdir, classlist, args)

print 'train_data.shape: ', train_data.shape, ' train_labels.shape: ', train_labels.shape
print 'valid_data.shape: ', valid_data.shape, ' valid_labels.shape: ', valid_labels.shape

'random shuffle training & validation data once'
d, l = rshuffle(train_data, train_labels)
dt, lt = rshuffle(valid_data, valid_labels)

'save to pickle file'
dstdir = os.path.join(args.datadir, 'pfiles')
if not os.path.exists(dstdir): 
    os.mkdir(dstdir)


for k in range(4): 
    imin = k*60000
    imax = (k+1)*60000    
    dfilename = 'train-images-%02d.pickle' % k
    dfile = os.path.join(dstdir, dfilename)
    with open(dfile, 'wb') as f:
        cPickle.dump(d[imin:imax], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    lfilename = 'train-labels-%02d.pickle' %k 
    lfile = os.path.join(dstdir, lfilename)
    with open(lfile, 'wb') as f:
        cPickle.dump(l[imin:imax], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

'save validation data/labels '
dtfile = os.path.join(dstdir, 'test-images.pickle')
with open(dtfile, 'wb') as f:
    cPickle.dump(dt, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
	
ltfile = os.path.join(dstdir, 'test-labels.pickle')
with open(ltfile, 'wb') as f:
    cPickle.dump(lt, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

