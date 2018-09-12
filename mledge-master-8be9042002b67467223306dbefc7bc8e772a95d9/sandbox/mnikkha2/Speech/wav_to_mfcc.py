import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import os
import cv2

def file2mfcc(fileName):
	(rate,sig) = wav.read(fileName)
	if len(sig) != 16000:
		return False,[]
	mfcc_feat = mfcc(sig,rate)	
	return True,mfcc_feat

Save_dir = 'Google_Data_Image/'
labelNames = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
dataFolder = 'Google_Speech_Data/'

for label in labelNames:
	currentLabelFolder = Save_dir+label+'/'
	if not os.path.exists(currentLabelFolder):
		os.makedirs(currentLabelFolder)
	print currentLabelFolder
	myFolder = dataFolder+label+'/'
	for fileName in os.listdir(myFolder):	
	#	m = file2mfcc('Google_Speech_Data/one/0a7c2a8d_nohash_0.wav')
		success, m = file2mfcc(myFolder+fileName)
		if success == False:
			continue
		m = m.reshape(m.shape[0],m.shape[1],1)
		m = m - m.min()
		m = m/m.max()
		m = (m*255).astype('uint8')
		print m.shape, m.max(), m.min()
		theImage = cv2.resize(m,(128,128))
		#print theImage.shape, theImage.min(), theImage.max()
		cv2.imwrite(currentLabelFolder+fileName.split('wav')[0]+'png', theImage)
		#cv2.imshow('img', theImage)
		#plt.pcolormesh(m)
		#plt.title(fileName)
		#plt.plot(m)
		#plt.show()
