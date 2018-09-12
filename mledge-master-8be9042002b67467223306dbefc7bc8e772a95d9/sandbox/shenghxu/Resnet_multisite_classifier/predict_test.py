
from resnet import *
from datetime import datetime
import time
from cifar10_input import *
from cifar10_train import *
from cifar10_train_2 import *
import numpy as np

#train1 = cifar10_train.Train()
train1 = Train()
train2 = Train_2()

#vali_data, vali_labels = read_validation_data()

#b1data,b1label = train1.generate_vali_batch(vali_data,vali_labels, FLAGS.validation_batch_size)

randimg = np.random.randn(128,32,32,3)
# testout = train1.predict_img(randimg,'./data_100/model.ckpt-50000')

testout2 = train2.predict_img(randimg,'./data_100_2/model.ckpt-10000')

print 'testout:',testout.shape
print 'testout2:',testout2.shape