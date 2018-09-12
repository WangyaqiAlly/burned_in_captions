import cv2
print 'cv2 ' + cv2.__version__
import sys
import os
import numpy as np
print 'numpy ' + np.__version__


def read_data():
	dataFolder = '../celebA/img_align_celeba/'
	labelsFile  = '../celebA/list_attr_celeba.txt'
	fileNames = os.listdir(dataFolder)
	fileNames.sort()

	#label_index = 31  #Smiling=1, non-smiling=-1
	#numTrain = 90000
	#numTest = 7000
	label_index = 39 #Young=1 or Old=-1
	numTrain = 40000
	numTest = 5000
	# index = 20 --> Male=1 Female=-1

	height = 64
	width = 64

	numFaces = 202599
	d = []
	counter = 0

	#numTrain = 97000   #45k old people only
	#numTest = 5000



	for f in fileNames:
	    if '.jpg' not in f:
		continue
	    counter += 1
	    if counter >numFaces:
		break
	    img = cv2.imread(dataFolder+f,cv2.IMREAD_COLOR)
	    img = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)
	    img = img.transpose(2,0,1)
	    d.append(img)


	d = np.array(d).reshape(len(d),3*width*height).astype('int32')
	#d = d/255.

	#print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()


	########### read the labels
	labels = np.zeros(shape = (202599,1,1, 40))
	with open(labelsFile, 'r') as f:
		i = 0
		for line in f:
			if i > 1:
				line = line.strip('\r\n')			
				#print len(line.split()[1:])
				labels[i-2,0,0,:] = line.split()[1:]
			i += 1
	labels = np.squeeze(labels)
	print labels.shape
	print "All the Young people:",np.sum((labels[:,label_index]+1)/2,axis=0)

	chosen = []
	counter_negative = 0
	counter_positive = 0
	i = 0



	while len(chosen) < numTrain + numTest :
		#if counter_negative < numTrain+numTest and labels[i,label_index] == -1:
		#	chosen.append(i) 
		#	counter_negative += 1

		if counter_positive < numTrain+ numTest and labels[i,label_index] == 1:
			chosen.append(i) 
			counter_positive += 1

		i += 1

	dTrain = d[chosen[:numTrain]]
	dTest = d[chosen[numTrain:numTrain+numTest]]
	trainLabelsTmp = (labels[chosen[:numTrain],label_index]+1)/2
	testLabelsTmp  = (labels[chosen[numTrain:numTrain+numTest],label_index]+1)/2
	trainLabelsTmp = trainLabelsTmp.reshape(numTrain).astype('int32')
	testLabelsTmp = testLabelsTmp.reshape(numTest).astype('int32')
	print dTrain.shape
	print trainLabelsTmp.shape
	
		
	#trainLabels = np.zeros(shape=(numTrain,2))
	#for k in xrange(numTrain):
	#	trainLabels[k][trainLabelsTmp[k]] = 1.
	
	#print trainLabels[:10]

	return dTrain,trainLabelsTmp

#read_data()


