import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

ON = 1
OFF = 2
switch = OFF 

foldNum = 9

retinaNameArr = ['random_50']


if switch == OFF:
    i = 0
    final_acc = []
    ratioLabels = []
    ratioPos = []
    modelRatio = []
    modelAcc = [] 
    for retinaName in retinaNameArr:
        i += 1
        k = 2
        images_num= [10000]
        fold_acc = [0.70949996]
        metaPath = './record/'+retinaName+'/retina_record_meta.txt'
        f = open(metaPath,'r')
        for line in f:
            data = line.split()
            wrongnum = int(data[1])
            correctnum = int(data[2])
            forwardnum = int(data[3])
            images_num += [images_num[-1] + forwardnum]
        f.close()

        while os.path.exists('./record/'+retinaName+'/retina_record_fold_'+str(k)+'.hdf'):
            retinaPath = './record/'+retinaName+'/retina_record_fold_'+str(k)+'.hdf'
            h5 = h5py.File(retinaPath, 'r')
            test_acc   = np.array(h5.get('test_acc'))
            total_acc  = 0.0
            for i in range(10):
                total_acc += test_acc[len(test_acc)-i-1]
            total_acc /= 10
            fold_acc += [total_acc] 
            k += 1
        modelAcc += [fold_acc]
        modelRatio += [images_num]

    f = open('text_result.txt', 'w')
    for i, retinaName in enumerate(retinaNameArr):
        f.write(retinaName+'x=[')
        print retinaName+'x=[',
        for j in range(foldNum):
            outputText = '%d' % modelRatio[i][j]
            print outputText,
            f.write(outputText+' ')
        f.write("]\n"+retinaName+'y=[')
        print ']'
        print retinaName+'y=[',
        for j in range(foldNum):
            outputText = '%.5lf' % modelAcc[i][j]
            f.write(outputText+' ')
            print outputText,
        f.write(']\n')
        print ']'
    f.close()


