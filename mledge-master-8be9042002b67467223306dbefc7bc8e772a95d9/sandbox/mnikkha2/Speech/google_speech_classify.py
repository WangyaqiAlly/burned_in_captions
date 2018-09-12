# CUDA_VISIBLE_DEVICES='0' python ali.py
import argparse
import struct
import time
import sys
import os
import numpy as np
print 'numpy ' + np.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print 'tensorflow ' + tf.__version__
import cv2
print 'cv2 ' + cv2.__version__
import wave
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from python_speech_features import mfcc




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=150, type=int)
parser.add_argument('--n', help='number of units per layer', default=32, type=int)
parser.add_argument('--lr', help='learning rate', default=0.01, type=float)
parser.add_argument('--batch', help='batch size', default=500, type=int)
parser.add_argument('--epochs', help='training epochs', default=100000, type=int)
parser.add_argument('--model', help='output model', default='model.proto.ali')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--step', help='step to start training', default=0, type=int)
parser.add_argument('--dataFolder', help='Train dataset folder', default='./Google_Speech_Data/')
#parser.add_argument('--testDataFolder', help='Test dataset folder', default='./TestData/')
args = parser.parse_args()
print args


alpha = 0.02
numTrainData = 4000
numTestData = 2000
numLabels = 10
labelNames = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

testAccRes = 0
trainAcc = 0


tag = sys.argv[0]
if tag.endswith('.py'):
    tag = tag[:-3]

if not os.path.exists(tag):
    os.makedirs(tag)




#### Reading and parsing training data
masterFolderName = args.dataFolder

dataFolderNames = os.listdir(masterFolderName)

maxNumFrames = 16000
d = []
SXX = []
labels = []
myMaxLen = 0
numFilesPerClass = {'zero':0, 'one':0, 'two':0}

for folderName in dataFolderNames:
  if folderName not in labelNames:
    continue
  fileNames = os.listdir(masterFolderName+folderName)
  for f in fileNames:
	#tmp1 = wave.open(masterFolderName+folderName+'/'+f,'r')
	#print tmp1.getparams()
	
	fs1, tmp = wav.read(masterFolderName+folderName+'/'+f)
	if len(tmp) != maxNumFrames:
		continue
	#numFilesPerClass[folderName] += 1
	#numFrames = tmp1.getnframes()
	#if numFrames>myMaxLen:
	#	myMaxLen = numFrames
	#plt.plot(tmp)
	#plt.title(f)
	#plt.show()
	#print tmp[0:10], len(tmp), tmp.shape, fs1, f
	#numFrames = tmp.getnframes()
	#if numFrames>maxNumFrames:
	#	maxNumFrames = numFrames
	#strFrames = tmp.readframes(numFrames)
	#frames = []
	#for i in range(numFrames):
	#	frames.append(float(int((strFrames[2*i+1:2*i+2]+strFrames[2*i:2*i+1]).encode('hex'),16)))
	#myTMP = frames
	myTMP = tmp
	#if len(myTMP)<maxNumFrames:
	#	myTMP = np.append(myTMP,np.zeros(maxNumFrames-len(myTMP)),axis=0)
		#myTMP.extend(np.zeros(maxNumFrames-len(myTMP)))
	#else:
	#	myTMP = myTMP[0:maxNumFrames]
	d.append(myTMP)
	#myMFCC = mfcc(d[-1],fs1)
	#sxx = myMFCC
	#plt.pcolormesh(sxx)
	#plt.show()
	#print myMFCC.shape, myMFCC[0]
	fs, tSeg, sxx = sig.spectrogram(d[-1],fs=16000,nperseg=200)
	#plt.pcolormesh(tSeg, fs,sxx)
	#plt.title(folderName)
	#plt.show()
	#print fs1
	SXX.append(sxx/sxx.max())
	labelInd = labelNames.index(folderName)

	#labelInd = int(f[0])
	labels.append(np.zeros(numLabels))
	labels[-1][labelInd] += 1
	#print f,labels[-1]
	#tmp1.close()


#print "Max file len:",myMaxLen
#hell = input('What to do?')


labels = np.array(labels).reshape(len(labels),numLabels).astype('float32')
SXX = np.array(SXX).reshape(len(SXX),sxx.shape[0],sxx.shape[1],1).astype('float32')
#SXX = SXX-SXX.min()
#SXX = SXX/SXX.max()
print SXX.shape,SXX.min(),SXX.max()

#print numFilesPerClass
print np.sum(labels[:,0]), np.sum(labels[:,1]), np.sum(labels[:,2])
#hell = input('What to do?')

ind = np.arange(SXX.shape[0])
#ind = np.random.permutation(SXX.shape[0])
np.random.shuffle(ind)
myData = SXX[ind]
myLabels = labels[ind]
print myData.shape, myData.min(), myData.max()
print myLabels.shape, myLabels.min(), myLabels.max()

dataTrain = myData[:numTrainData]
dataTest = myData[numTrainData:numTrainData+numTestData]
labelsTrain = myLabels[:numTrainData]
labelsTest = myLabels[numTrainData:numTrainData+numTestData]

print 'dataTrain.shape',dataTrain.shape, 'dataTrain.min()',dataTrain.min(),'dataTrain.max()',dataTrain.max()
print 'labelsTrain.shape',labelsTrain.shape
print 'dataTest.shape',dataTest.shape, 'dataTest.min()',dataTest.min(),'dataTest.max()',dataTest.max()
print 'labelsTest.shape',labelsTest.shape

print np.sum(labelsTrain[:,0]),np.sum(labelsTrain[:,1]),np.sum(labelsTrain[:,2])
print np.sum(labelsTest[:,0]),np.sum(labelsTest[:,1]),np.sum(labelsTest[:,2])
#myHello = input('Hello?')

#######		
'''
#### Reading and parsing testing data
folderName = args.testDataFolder
fileNames = os.listdir(folderName)

dTest = []
SXXTest = []
labels_test = []
for f in fileNames:
	if 'nicolas' in f:# or 'theo' in f:
		continue
	fs1, tmp = wav.read(folderName+f)
	if 'theo' in f:
		tmp = (tmp)*10
	#plt.plot(tmp)
	#plt.title(f)
	#plt.show()
	#numFrames = tmp.getnframes()
	#strFrames = tmp.readframes(numFrames)
	#frames = []
	#for i in range(numFrames):
	#	frames.append(float(int(strFrames[2*i:2*i+2].encode('hex'),16)))
	#myTMP=frames
	myTMP = tmp
	if len(myTMP)<maxNumFrames:
		#myTMP.extend(np.zeros(maxNumFrames-len(myTMP)))
		myTMP = np.append(myTMP,np.zeros(maxNumFrames-len(myTMP)),axis=0)
	else:
		myTMP = myTMP[0:maxNumFrames]
	dTest.append(np.array(myTMP)/max(myTMP))
	myMFCC = mfcc(dTest[-1],fs1)
	sxx = myMFCC
	#fs1, tSeg, sxx = sig.spectrogram(dTest[-1])
	#print fs1
	#plt.pcolormesh(tSeg, fs1, sxx)
	#plt.show()
	SXXTest.append(sxx)

	labelInd = int(f[0])
	labels_test.append(np.zeros(numLabels))
	labels_test[-1][labelInd] += 1
	#tmp.close()

#maxNumFrames= 18400  ## 115*160 , 160 frames (20ms) is what people usually use for audio for 8Khz
#maxNumFrames= 8000  ## 50*160 , 160 frames (20ms) is what people usually use for audio for 8Khz

#for i in range(len(dTest)):
#	if len(dTest[i])<maxNumFrames:
#		dTest[i].extend(np.zeros(maxNumFrames-len(dTest[i])))
#	else:
#
#		dTest[i] = dTest[i][0:maxNumFrames]


labels_test = np.array(labels_test[:numTestData]).reshape(numTestData,10).astype('float32')
SXXTest = np.array(SXXTest[:numTestData]).reshape(numTestData,sxx.shape[0],sxx.shape[1],1).astype('float32')
#dTest = np.array(dTest[:numTestData]).reshape(numTestData,maxNumFrames).astype('float32')
SXXTest = SXXTest/SXXTest.max()

print 'SXXTest.shape',SXXTest.shape, 'SXXTest.min()',SXXTest.min(),'SXXTest.max()',SXXTest.max()
print 'labels_test.shape',labels_test.shape

#######'''




def classifier(args, x, reuse=None):
	with tf.variable_scope('C',reuse=reuse):
		# 101 X 91 X 1 -> 97 X 87 X 1
		e = tf.layers.conv2d(inputs=x, filters=5, kernel_size=5, strides=1,data_format='channels_last',activation=tf.nn.relu, padding='valid') ; print e
		#e = tf.maximum(alpha*e,e) # LeakyReLU
		e = tf.image.resize_bilinear(images=e,size=[80,80]) ; print e
		# 80 X 80 X 1 -> 36 X 36 X 1
		e = tf.layers.conv2d(inputs=e, filters=5, kernel_size=10, strides=2,data_format='channels_last',activation=tf.nn.relu, padding='valid') ; print e
		#e = tf.maximum(alpha*e,e) # LeakyReLU
		e = tf.image.resize_bilinear(images=e,size=[20,20]) ; print e
		# 10 X 10 X 1 -> 2 X 2 X 1
		#e = tf.layers.conv2d(inputs=e, filters=1, kernel_size=9, strides=1,data_format='channels_last',activation=None, padding='valid') ; print e
		# 20 X 20 X 1 -> 6 X 6 X 1
		e = tf.layers.conv2d(inputs=e, filters=1, kernel_size=10, strides=2,data_format='channels_last',activation=tf.nn.relu, padding='valid') ; print e
		#e = tf.maximum(alpha*e,e) # LeakyReLU
		e = tf.image.resize_bilinear(images=e,size=[4,4]) ; print e
		#e = tf.layers.conv2d(inputs=e, filters=1, kernel_size=7, strides=1,data_format='channels_last',activation=tf.nn.relu, padding='valid') ; print e
		e = tf.reshape(e,[-1,16]); print e
		e = tf.layers.dense(inputs=e, units=numLabels, activation=None, name="myDense1",reuse=reuse); print e
		e = tf.identity(e,name='cout') ; print e
		return e





'''def classifier(args, x, reuse=None):
	with tf.variable_scope('C',reuse=reuse):
		# 129 X 35 X 1 -> 125 X 31 X 64
		e = tf.layers.conv2d(inputs=x, filters=10, kernel_size=5, strides=1,data_format='channels_last',activation=None, padding='valid') ; print e
		e = tf.maximum(alpha*e,e) # LeakyReLU
		# 125 X 31 X 64 -> 61 X 14 X 128
		e = tf.layers.conv2d(inputs=e, filters=10, kernel_size=5, strides=2,data_format='channels_last',activation=None, padding='valid') ; print e
		e = tf.maximum(alpha*e,e) # LeakyReLU
		# 61 X 14 X 128 -> 29 X 5 X 256
		e = tf.layers.conv2d(inputs=e, filters=5, kernel_size=5, strides=2,data_format='channels_last',activation=None, padding='valid') ; print e
		e = tf.maximum(alpha*e,e) # LeakyReLU
		# 29 X 5 X 256 -> 13 X 1 X 256
		e = tf.layers.conv2d(inputs=e, filters=1, kernel_size=5, strides=2,data_format='channels_last',activation=None, padding='valid') ; print e
		e = tf.maximum(alpha*e,e) # LeakyReLU
		e = tf.reshape(e,[-1,13]); print e
		e = tf.layers.dense(inputs=e, units=10, activation=None, name="myDense1",reuse=reuse); print e
	        e = tf.identity(e,name='cout') ; print e
		return e'''



myInputShape = dataTrain.shape
x = tf.placeholder('float32', [None,myInputShape[1],myInputShape[2],myInputShape[3]],name='x'); print x
label = tf.placeholder('float32',[None, numLabels]); print label





class_out = classifier(args, x, reuse=False)
dloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=class_out,labels=label)); print dloss

dopt = tf.train.AdamOptimizer(learning_rate=args.lr)
#dopt = tf.train.RMSPropOptimizer(learning_rate=args.lr).minimize(dloss)
v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'C')
dgrads = dopt.compute_gradients(dloss,var_list=v1)
dtrain = dopt.apply_gradients(dgrads)
dnorm = tf.global_norm([i[0] for i in dgrads])


#print "class_out.shape= {0} and label.shape={1}".format(class_out.shape, label.shape)
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(class_out), axis=1), tf.argmax(label, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver(max_to_keep = 10000)
step = args.step

with tf.Session() as sess:
	if step == 0:
		sess.run(tf.global_variables_initializer())
	else:
		model_path = os.path.join(tag, 'model-' + str(step))
		saver.restore(sess, model_path)

	for i in range(args.epochs):
    	# shuffle both the data and the labels
		ind = np.arange(dataTrain.shape[0])
		np.random.shuffle(ind)
		dataTrain = dataTrain[ind]
		labelsTrain = labelsTrain[ind]
		dc=0.
		dn=0.
		acc = 0.
		t=0.
		for j in range(0,dataTrain.shape[0],args.batch):
			input_x = dataTrain[j:j+args.batch]
			input_y = labelsTrain[j:j+args.batch]
			_,dc_,dn_,acc_ = sess.run([dtrain,dloss,dnorm, accuracy], feed_dict={x:input_x,label:input_y})
			####myOut =  sess.run([spectrogram],feed_dict={x:input_x, window:hanning(N)})
			#_, dc_, pred = sess.run([dopt, dloss, correct_prediction], feed_dict={x:input_x,label:input_y})
			dc += dc_
			dn += dn_
			acc += acc_
			t += 1.
			step += 1
			if (step % 5000) == 0:
				savepath = saver.save(sess, os.path.join(tag, 'model'), global_step=step, write_meta_graph=False)
				print 'saving ',savepath
			#if j%500==0:
		trainSetAcc = acc/t
		dcSummary = dc/t
		#print 'epoch',i,'Training cost',dc/t,'dnorm',dn/t, 'Train Set Accuracy:', trainSetAcc , 'Test Set Accuracy:', testAccRes

		
		t = 0.
		testAccRes = 0.
		tmpAcc = 0
		for k in range(0,dataTest.shape[0],args.batch):
			input_x_test = dataTest[k:k+args.batch]
			input_y_test = labelsTest[k:k+args.batch]
			tmpAcc = sess.run(accuracy, feed_dict={x:input_x_test, label:input_y_test})
			#print tmpAcc, input_y_test[100]
			testAccRes += tmpAcc
			t += 1.
		testAccRes = testAccRes/t
		print 'epoch',i,'Training cost',dcSummary , 'Train Set Accuracy:', trainSetAcc , 'Test Set Accuracy:', testAccRes
		
				#testAcc = []
				#for kk in range(0, dTest.shape[0], args.batch):
				#	tmpPred= sess.run(correct_prediction, feed_dict={x:dTest[kk:kk+args.batch], label:labels_test[kk:kk+args.batch]})
					#print tmpPred
				#	testAcc.extend(tmpPred)
				#testAccRes = sess.run(tf.reduce_mean(tf.cast(tf.convert_to_tensor(testAcc), tf.float32)))
				#print "\nAccuracy on test data=",testAccRes
				#print "Accuracy on test data=", sess.run(accuracy, feed_dict={x:dTest[kk:kk+args.batch], label:labels_test[kk:kk+args.batch]})

		# write model, redirect stderr to supress annoying messages
		#with open(os.devnull, 'w') as sys.stdout:
		#graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['G/gout', 'D/dout'])
		#sys.stdout=sys.__stdout__
		#tf.train.write_graph(graph, '.', args.model, as_text=False)



