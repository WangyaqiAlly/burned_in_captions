'''
    gen_imgnet+.py

    generate 'imgnet+' dataset from 1000-class
    ImageNet dataset 

    Each "super class" in the imgnet+ class
    is created by merging images across
    multiple sub-classes from the original ImageNet
    

'''


import os
from shutil import copyfile
import numpy as np
import csv
import argparse
import struct
import cv2

'populate class list'
classlist = []
classlist.append('invertebrate')
classlist.append('bird')
classlist.append('vehicle')
classlist.append('dog')
classlist.append('clothing')
# ----------------- #
classlist.append('container')
classlist.append('construction')
classlist.append('device')
classlist.append('fish')
classlist.append('food')
classlist.append('horse')
classlist.append('music')
classlist.append('plant')
classlist.append('person')
classlist.append('room')

def csv2dict(args):

    'lut from imgfile to bbox'
    img2bbox = {}
    bboxfile = os.path.join(args.srcdir, args.bboxfile)
    if not os.path.exists(bboxfile): 
	print 'missing ', bboxfile
    else:
	print 'found ', bboxfile

        with open(bboxfile, 'r') as csvfile:
	    creader = csv.reader(csvfile)
	    # for i in range(10):
	    for row in creader: 
    		# row = creader.next()
            	imgfile = row[0]
            	xmin = float(row[1]) 
		ymin = float(row[2])
	        xmax = float(row[3]) 
		ymax = float(row[4])
	    	img2bbox[imgfile] =  (xmin, ymin, xmax, ymax) 

    return img2bbox

def convert(imgfile, srcdir,  dstdir, bbox): 

    print '-'*10, ' converting ', '-'*10

    xmin, ymin, xmax, ymax = bbox
    srcfile = os.path.join(srcdir, imgfile)
    dstfile = os.path.join(dstdir, imgfile[:-4]+'bmp')
#    dstfile = os.path.join(dstdir, imgfile[:-4]+'jpg')
    print 'srcfile: ', srcfile
    print 'dstfile: ', dstfile

    im = cv2.imread(srcfile)
    h, w, d = im.shape

    x0 = 0.5*(xmin+xmax);
    y0 = 0.5*(ymin+ymax);
    wcrop = (xmax-xmin)*w
    hcrop = (ymax-ymin)*h
    print 'bbox: ', bbox
    print '%d x %d => %.2f x %.2f' % (w, h, wcrop, hcrop)
    if wcrop>hcrop:
	if wcrop<h: 
	    print 'landscape, expand bbox: ', imgfile
	    if y0*h>wcrop*0.5 and y0*h+wcrop*0.5<h:
		ymin = y0-wcrop*0.5/float(h)
		ymax = y0+wcrop*0.5/float(h)
	    elif ymin<1-ymax: 
		ymin = 0.0
		ymax = wcrop/float(h)
	    else: 
		ymin = 1-wcrop/float(h)
		ymax = 1.0
	else: 
	    print 'landscape, crop bbox: ', imgfile 
	    ymin = 0.0
	    ymax = 1.0
	    xmin = x0-h*0.5/float(w)
	    xmax = x0+h*0.5/float(w)
    elif hcrop>wcrop: 
	if hcrop<=w: 
	    print 'portrait, expand bbox: ', imgfile
	    if x0*w>hcrop*0.5 and x0*w+hcrop*0.5<w: 
		xmin = x0-hcrop*0.5/float(w)
		xmax = x0+hcrop*0.5/float(w)
	    elif xmin<1-xmax: 
		xmin = 0.0
		xmax = hcrop/float(w)
	    else: 
		xmin = 1-hcrop/float(w)
		xmax = 1.0
	else: 
	    print 'portrait, crop bbox: ', imgfile
	    xmin = 0.0
	    xmax = 1.0
	    ymin = y0-w*0.5/float(h)
	    ymax = y0+w*0.5/float(h)
    else: 
	print 'perfectly square: ', imgfile
	print 'wcrop: ', wcrop, ',  hcrop: ', hcrop 

    print 'resizing ... determine resize ratio'
    print 'xmin=', xmin, ' | xmax=', xmax, ' | ymin=', ymin, ' | ymax=', ymax
    wcrop = (xmax-xmin)*w
    hcrop = (ymax-ymin)*h
    icrop = max(wcrop, hcrop)
    print 'target image size: wcrop=', wcrop, ',  hcrop=', hcrop 
    ratio = float(args.n)/float(icrop)
    wnew = int(np.ceil(w*ratio))
    hnew = int(np.ceil(h*ratio))
    im2 = cv2.resize(im, (wnew, hnew), interpolation = cv2.INTER_CUBIC) 
    print 'resizing ratio: ', ratio, '  resized image: ', im2.shape

    'cropping ...'
#    n0 = int(args.n/2)
#     x0 = 0.5*(xmin+xmax); w0 = int(wnew*x0)
#     y0 = 0.5*(ymin+ymax); h0 = int(hnew*y0)
    wmin = int(np.floor(wnew*xmin)); # max(w0-n0, 0)
    wmax =  min(wmin+args.n, wnew)
    hmin = int(np.floor(hnew*ymin)); # max(h0-n0, 0)	
    hmax = min(hmin+args.n, hnew)
    im3 = im2[hmin:hmax, wmin:wmax]
    print 'xmin=', xmin, ' | xmax=', xmax, ' | ymin=', ymin, ' | ymax=', ymax
    print 'wmin=', wmin, ' | wmax=', wmax, ' | hmin=', hmin, ' | hmax=', hmax
    print 'cropped image: ', im3.shape

    print imgfile, 
    if (im3.shape[1]>im3.shape[0]):
	print '  landscape crop, need to expand h'
	hdiff = args.n-im3.shape[0]
	hpad0 = int(hdiff/2)
	hpad1=hdiff-hpad0
	imnew = np.lib.pad(im3, ((hpad0, hpad1), (0, 0), (0, 0)), 'edge')
    elif (im3.shape[0]>im3.shape[1]): 
	print '  portrait crop, need to expand w'
	wdiff = args.n-im3.shape[1]
	wpad0 = int(wdiff/2)
	wpad1=wdiff-wpad0
	imnew = np.lib.pad(im3, ((0, 0), (wpad0, wpad1), (0, 0)), 'edge')
    else: 
	print '  perfectly square already'
	imnew = im3

    print 'final image: ', dstfile, ' shape ', imnew.shape

    'save to dstfile'
    cv2.imwrite(dstfile, imnew)

###### start main program ####### 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nclass', help='number of target super classes', default=5, type=int)
parser.add_argument('--srcdir', help='source directory of original ImageNet dataset',
		    		default='/home/xiaoqzhu/dataset/imagenet-15')
parser.add_argument('--dstdir', help='destination directory of original ImageNet dataset',
		    		default='/home/xiaoqzhu/dataset/imagenet-plus-5-class-256')
parser.add_argument('--bboxfile', help='csv file containing bounding box info', 
				default='imagenet_2012_bounding_boxes.csv')
parser.add_argument('--n', help='target resolution', default=256, type=int)


args = parser.parse_args()
print args

img2bbox = csv2dict(args)
# print img2bbox
print 'total # of images:', len(img2bbox)

traindir = os.path.join(args.dstdir, 'train')
validdir = os.path.join(args.dstdir, 'validation')
if not os.path.exists(traindir): 
    os.mkdir(traindir)

if not os.path.exists(validdir): 
    os.mkdir(validdir)

for i in range(args.nclass):
    classname = classlist[i]
    classdir = os.path.join(args.srcdir, classname)
    imglist = os.listdir(classdir)
    classtag = '%02d-%s' % (i, classname)

    dst_validdir = os.path.join(validdir, classtag)
    if not os.path.exists(dst_validdir): 
	os.mkdir(dst_validdir)

    dst_traindir = os.path.join(traindir, classtag)
    if not os.path.exists(dst_traindir): 
   	os.mkdir(dst_traindir)

    print 'class %s: %d images' % (classname, len(imglist))
    for (j,imgfile) in enumerate(imglist): 
	if j % 5 == 0:
	    targetdir = dst_validdir
	else: 
	    targetdir = dst_traindir
       	
	bbox = 0., 0., 1., 1.
	if imgfile in img2bbox: 
	    bbox = img2bbox[imgfile]
	
#	if j < 50: 
	convert(imgfile, classdir, targetdir, bbox)


