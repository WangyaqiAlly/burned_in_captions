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

    xmin, ymin, xmax, ymax = bbox
    srcfile = os.path.join(srcdir, imgfile)
    dstfile = os.path.join(dstdir, imgfile[:-4]+'bmp')
    print 'srcfile: ', srcfile
    print 'dstfile: ', dstfile

    im = cv2.imread(srcfile)
    h, w, d = im.shape

    wcrop = (xmax-xmin)*w
    hcrop = (ymax-ymin)*h
    icrop = max(wcrop, hcrop)
    print 'bbox: ', bbox
    print '%d x %d => %.2f x %.2f' % (w, h, wcrop, hcrop)

    'resizing ... determine resize ratio'
    ratio = float(args.n)/float(icrop)
    wnew = int(np.ceil(w*ratio))
    hnew = int(np.ceil(h*ratio))
    im2 = cv2.resize(im, (wnew, hnew), interpolation = cv2.INTER_AREA) 
    print 'resizing ratio: ', ratio, '  resized image: ', im2.shape

    'cropping ...'
    n0 = int(args.n/2)
    x0 = 0.5*(xmin+xmax); w0 = int(wnew*x0)
    y0 = 0.5*(ymin+ymax); h0 = int(hnew*y0)
    wmin = max(w0-n0, 0)
    wmax = min(wmin+args.n, wnew)
    hmin = max(h0-n0, 0)	
    hmax = min(hmin+args.n, hnew)
    im3 = im2[hmin:hmax, wmin:wmax]
    print 'cropped image: ', im3.shape

    if (im3.shape[1]>im3.shape[0]):
	print 'landscape crop, need to expand h'
	hdiff = args.n-im3.shape[0]
	hpad0 = int(hdiff/2)
	hpad1=hdiff-hpad0
	imnew = np.lib.pad(im3, ((hpad0, hpad1), (0, 0), (0, 0)), 'edge')
    elif (im3.shape[0]>im3.shape[1]): 
	print 'portrait crop, need to expand w'
	wdiff = args.n-im3.shape[1]
	wpad0 = int(wdiff/2)
	wpad1=wdiff-wpad0
	imnew = np.lib.pad(im3, ((0, 0), (wpad0, wpad1), (0, 0)), 'edge')
    else: 
	print 'perfectly square already'
	imnew = im3

    print 'final image: ', dstfile, ' shape ', imnew.shape

    'save to dstfile'
    cv2.imwrite(dstfile, imnew)

###### start main program ####### 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--srcdir', help='source directory',
		    		default='/home/xiaoqzhu/dataset/imagenet-15/clothing')
parser.add_argument('--dstdir', help='destination directory',
		    		default='/home/xiaoqzhu/dataset/imagenet-plus-5-class/train/04-clothing')
parser.add_argument('--imgfile', help='source image file', 
				default='n04259630_1860.JPEG')

parser.add_argument('--n', help='target resolution', default=64, type=int)


args = parser.parse_args()
print args

bbox = 0., 0., 1., 1.	
convert(args.imgfile, args.srcdir, args.dstdir, bbox)
