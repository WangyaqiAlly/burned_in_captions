import argparse
from retinaNode import RetinaNode
from retinaUpload import Uploader
import matplotlib.pyplot as plt
import os
import h5py

UploadNum = 10
record_prefix = 'retina_record'
record_dir = './record'
retinaPath = os.path.join(record_dir, record_prefix)
foldnameArr = ['fold_2.hdf', 'fold_3.hdf', 'fold_4.hdf', 'fold_5.hdf', 'fold_6.hdf', 'fold_7.hdf', 'fold_8.hdf', 'fold_9.hdf', 'fold_10.hdf', 'fold_11.hdf']

def save_parameters(filename, train_loss, train_acc, test_loss, test_acc):
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('train_loss', data=train_loss, dtype='f')
    h5.create_dataset('train_acc',  data=train_acc,  dtype='f')
    h5.create_dataset('test_loss',  data=test_loss,  dtype='f')
    h5.create_dataset('test_acc',   data=test_acc,   dtype='f')
    h5.close()
    
def write_txt(filename, train_loss, train_acc, test_loss, test_acc):
    f = open(filename, 'w')
    for i in range(len(train_loss)):
        f.write('%.6lf %.6lf %.6lf %.6lf\n' % (train_loss[i], train_acc[i], test_loss[i], test_acc[i]))
    f.close()

def update_metadata(filename, foldnum, wrongnum, correctnum, forwardnum):
    f = open(filename, 'a+')
    f.write('%d %d %d %d\n' % (foldnum, wrongnum, correctnum, forwardnum))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="the name of node")
    parser.add_argument('--batch_num', help="the batch size, default is 128", default=128, type=int)
    parser.add_argument('--epoch', help="the number of epoch to train", default=2000, type=int)
    parser.add_argument('--record_name', help='the name of record', default=None)
    parser.add_argument('--rate', help='the ratio of selection from correct', default=1.0, type=float)
    parser.add_argument('--mode', default=0, type=int)

    args = parser.parse_args()
    print args
    assert args.name != None, ("name cannot be none")
    if args.record_name is not None:
        retinaPath = os.path.join(record_dir, args.record_name)
        if os.path.exists(retinaPath) == False:
            os.mkdir(retinaPath)
        retinaPath = os.path.join(retinaPath, record_prefix)
        print 'retinaPath', retinaPath

    obj = RetinaNode(args.name, args.record_name, args.batch_num)
    obj.initialize()
    if args.mode == 0:
        trainlossArr, trainaccArr, testlossArr, testaccArr = obj.train(args.epoch)
        save_parameters(retinaPath + '_fold_' + str(1) + '.hdf',
                        trainlossArr, trainaccArr, 
                        testlossArr,  testaccArr)
        write_txt(retinaPath + '_fold_' + str(1) + '.txt',
                trainlossArr, trainaccArr, 
                testlossArr,  testaccArr)
        obj.save_model()
        print 'init neural network model finished'
        exit()
    else:
        obj.restore_model()

    uploadObj = Uploader('node_edge','node_central', args.record_name, args.record_name)
    for i in range(UploadNum):
        wrongnum, correctnum, forwardnum = uploadObj.filt_and_transfer(foldnameArr[i], obj, rate=args.rate, mode=args.mode)
        obj.updateReader()

        newtrainlossArr, newtrainaccArr, newtestlossArr, newtestaccArr = obj.train(args.epoch)
        save_parameters(retinaPath + '_fold_' + str(i+2) + '.hdf',
                        newtrainlossArr, newtrainaccArr, 
                        newtestlossArr,  newtestaccArr)

        write_txt(retinaPath + '_fold_' + str(i+2) + '.txt',
                newtrainlossArr, newtrainaccArr, 
                newtestlossArr,  newtestaccArr)

        update_metadata(retinaPath+'_meta.txt', i+2, wrongnum, correctnum, forwardnum)
