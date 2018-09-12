import os
from random import shuffle
import cv2
import tensorflow as tf
import numpy as np



def readSpeechImages():
	dataFolder = '../Speech/Google_Data_Image/'
	labelNames = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
	width, height = 128, 128
	labeledPercent = 10
	testPercent = 10
	numClasses = len(labelNames)

	'''if os.path.exists(dataFolder+'_unlabeled/'):
		deleteCommand = 'rm -r ' + dataFolder+'_unlabeled/'
		os.system(deleteCommand)
		os.makedirs(dataFolder+'_unlabeled/')
	else:
		os.makedirs(dataFolder+'_unlabeled/')'''

	test_images = []
	test_labels = []
	train_images_labeled = []
	train_labels = []
	train_images_unlabeled = []
	train_labels_unlabeled = []
	testData = []
	trainLabeledData = []
	trainUnlabeledData = []
	for indLabel,label in enumerate(labelNames):
		currentLabelFolder = dataFolder+label+'/'
	
		'''if os.path.exists(dataFolder+label+'_labeled/'):
			deleteCommand = 'rm -r '+ dataFolder+label+'_labeled/'
			os.system(deleteCommand)
			os.makedirs(dataFolder+label+'_labeled/')
		else:
			os.makedirs(dataFolder+label+'_labeled/')'''
		fileNames = os.listdir(currentLabelFolder)
		shuffle(fileNames)
	
		### 10% for testing
		for i in range(len(fileNames)//testPercent):
			#print currentLabelFolder+fileNames[i]
			img = cv2.imread(currentLabelFolder+fileNames[i],cv2.IMREAD_GRAYSCALE).reshape(width, height,1)
			#print img.shape
			#img = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)
			#test_images.append(tf.convert_to_tensor(img, dtype=tf.float32))
			test_images.append(img)
			test_labels.append(np.zeros(numClasses))
			test_labels[-1][indLabel] += 1
			#test_labels.append(tf.one_hot(tf.cast(indLabel, tf.int32), numClasses))
			#testData.append((test_images[-1],test_labels[-1]))
	
		### 10% Labeled for training
		for i in range(len(fileNames)//testPercent,(len(fileNames)//testPercent)+(len(fileNames)//labeledPercent)):
			img = cv2.imread(currentLabelFolder+fileNames[i],cv2.IMREAD_GRAYSCALE).reshape(width, height,1)
			#train_images_labeled.append(tf.convert_to_tensor(img, dtype=tf.float32))
			train_images_labeled.append(img)
			train_labels.append(np.zeros(numClasses))
			train_labels[-1][indLabel] += 1
			#train_labels.append(tf.one_hot(tf.cast(indLabel, tf.int32), numClasses))
			#trainLabeledData.append((train_images_labeled[-1],train_labels[-1]))
	
		### The rest is unlabeled
		for i in range(len(fileNames)//5,len(fileNames)):
			img = cv2.imread(currentLabelFolder+fileNames[i],cv2.IMREAD_GRAYSCALE).reshape(width, height,1)
			train_images_unlabeled.append(img)
			#train_images_unlabeled.append(tf.convert_to_tensor(img, dtype=tf.float32))
			#train_labels_unlabeled.append(tf.convert_to_tensor(0,tf.int32))
			#trainUnlabeledData.append((train_images_unlabeled[-1],train_labels_unlabeled[-1]))
	
	test_images = np.array(test_images).reshape(len(test_images),width,height,1).astype('float32')
	test_images = test_images/test_images.max()
	test_labels = np.array(test_labels).reshape(len(test_labels),numClasses).astype('int32')
	
	# shuffle both the data and the labels
	#ind = np.arange(test_images.shape[0])
	#np.random.shuffle(ind)
	#test_images = test_images[ind]
	#test_labels = test_labels[ind]
	#test_labels = tf.one_hot(tf.cast(test_labels, tf.int32), numClasses)		

	train_images_labeled = np.array(train_images_labeled).reshape(len(train_images_labeled),width,height,1).astype('float32')
	train_images_labeled = train_images_labeled/train_images_labeled.max()
	train_labels = np.array(train_labels).reshape(len(train_labels),numClasses).astype('int32')
	# shuffle both the data and the labels
	#ind = np.arange(train_images_labeled.shape[0])
	#np.random.shuffle(ind)
	#train_images_labeled = train_images_labeled[ind]
	#train_labels = train_labels[ind]
	#train_labels = tf.one_hot(tf.cast(train_labels, tf.int32), numClasses)

	train_images_unlabeled = np.array(train_images_unlabeled).reshape(len(train_images_unlabeled),width,height,1).astype('float32')
	train_images_unlabeled = train_images_unlabeled/train_images_unlabeled.max()
	# shuffle both the data and the labels
	#ind = np.arange(train_images_labeled.shape[0])
	#np.random.shuffle(ind)
	#train_images_unlabeled = train_images_unlabeled[ind]
	
	return train_images_labeled, train_labels, train_images_unlabeled, test_images, test_labels
	#return trainLabeledData, trainUnlabeledData, testData


	
