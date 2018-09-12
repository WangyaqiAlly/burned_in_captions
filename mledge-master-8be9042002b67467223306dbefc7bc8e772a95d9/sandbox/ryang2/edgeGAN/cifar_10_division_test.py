import argparse
import cPickle
import os
import numpy as np
import cv2

def unpickle(file):
    print 'Loading from: ', file
    with open(file, 'rb') as fin:
        dict = cPickle.load(fin)
    return dict

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dest_dir', required=True, help='Destination root directory of CIFAR_10')
parser.add_argument('-ns', '--num_site', default=2, help='Number of sites')
parser.add_argument('-nt', '--num_time', default=3, help='Number of time slices')
args = parser.parse_args()

data= []
label=[]

if not os.path.exists(args.dest_dir):
	print 'No input directory'
	exit()
    
fileNames = os.listdir(args.dest_dir+'/train/site-2/time-03/')
print fileNames
for f in fileNames:
    tempDict = unpickle(args.dest_dir+'/train/site-2/time-03/'+f)
    tempArr = tempDict['data']
    tempLabel = tempDict['labels']
    print min(tempLabel), max(tempLabel)
    print 'len of tempLabel: ', len(tempLabel), '   type of tempArr: ', tempArr.shape

    for i in range(len(tempLabel)):
      if tempLabel[i]!=10000:
        #d.append(tempArr[i])
        im = np.reshape(tempArr[i],(32,32,3),order='F')
        rows,cols,_ = im.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        im = cv2.warpAffine(im,M,(cols,rows))
        #gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #print gray_image.shape
        tempOneHot = np.eye(10)[tempLabel[i]]
        #tempOneHot[tempLabel[i]]=1

        #tempOneHot
        data.append(im)
        label.append(tempOneHot)

    print 'data shape: ', len(data), 'label shape:', len(label)
