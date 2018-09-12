import os
from random import shuffle

def createSpeechData():
	labeledPercent = 50*(0.9)/100.
	testPercent = 10
	testDataFolderOrig = './SpeechData10Test10Labeled/testData/'
	
	dataFolder = '../Speech/Google_Data_Image/'
	labelNames = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
	unlabeledDataFolder = './SpeechData10Test50Labeled/unlabeledData/'
	if not os.path.exists(unlabeledDataFolder):
		os.makedirs(unlabeledDataFolder)
	for indLabel,label in enumerate(labelNames):
		currentTestDataOrigFolder = testDataFolderOrig+label+'_labeled/'
		allTestFiles = os.listdir(currentTestDataOrigFolder)
		labeledDataFolder = './SpeechData10Test50Labeled/labeledData/'+label+'_labeled/'
		testDataFolder = './SpeechData10Test50Labeled/testData/'+label+'_labeled/'
		currentLabelFolder = dataFolder+label+'/'
		#print currentLabelFolder
		fileNames = os.listdir(currentLabelFolder)
		print len(fileNames)
		shuffle(fileNames)
		if not os.path.exists(labeledDataFolder):
			os.makedirs(labeledDataFolder)
			labeledCounter = 0
			i = 0
			while labeledCounter< int(labeledPercent*len(fileNames)):
				if fileNames[i] in allTestFiles:
					i +=1
					continue
				copyCommand = 'cp '+currentLabelFolder+fileNames[i] + ' '+ labeledDataFolder
				os.system(copyCommand)
				i += 1
				labeledCounter += 1
		print i, labeledCounter
		if not os.path.exists(testDataFolder):
			os.makedirs(testDataFolder)
			for fName in allTestFiles:
				copyCommand = 'cp '+currentTestDataOrigFolder+fName+ ' '+ testDataFolder
				os.system(copyCommand)
		while i < len(fileNames):
			if fileNames[i] in allTestFiles:
				i +=1
				continue
			copyCommand = 'cp '+currentLabelFolder+fileNames[i] + ' '+ unlabeledDataFolder+str(indLabel)+'_'+fileNames[i]
			os.system(copyCommand)
			i += 1
		

createSpeechData()
