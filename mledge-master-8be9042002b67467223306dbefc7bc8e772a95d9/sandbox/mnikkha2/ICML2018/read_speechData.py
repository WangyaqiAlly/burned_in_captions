import cv2
print 'cv2 ' + cv2.__version__
import sys
import os
import numpy as np
print 'numpy ' + np.__version__


def read_data():
	dataFolder = './SpeechData10Test50Labeled/unlabeledData/'
	fileNames = os.listdir(dataFolder)
	fileNames.sort()

	height = 128
	width = 128

	numImages = 17200
	d = []
	counter = 0



	for f in fileNames:
	    if '.png' not in f:
		continue
	    counter += 1
	    if counter > numImages:
		break
	    img = cv2.imread(dataFolder+f,cv2.IMREAD_GRAYSCALE)
	    img = img.reshape(width,height,1)
	    #img = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)
	    img = img.transpose(2,0,1)
	    
	    d.append(img)


	d = np.array(d).reshape(len(d),1*width*height).astype('int32')
	#d = d/255.

	#print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()


	return d, np.zeros(len(d))

#read_data()


