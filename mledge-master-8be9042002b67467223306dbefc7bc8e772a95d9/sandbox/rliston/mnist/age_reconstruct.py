# CUDA_VISIBLE_DEVICES='0' python age_reconstruct.py
import argparse
import struct
import time
import subprocess
import numpy as np
print 'numpy ' + np.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print 'tensorflow ' + tf.__version__
import cv2
print 'cv2 ' + cv2.__version__
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=10, type=int)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--model', help='output model', default='model.proto')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

with open('t10k-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    dt = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    dt = dt/255.

with open('t10k-labels-idx1-ubyte','rb') as f:
    h = struct.unpack('>II',f.read(8))
    lt = np.fromstring(f.read(), dtype=np.uint8).astype('int32')

print 'dt.shape',dt.shape,'lt.shape',lt.shape

with tf.Session() as sess:
    with open(args.model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    while True:
        np.random.shuffle(dt)
        # run a batch of MNIST images through the encoder to generate a batch of latent vectors
        z = sess.run('enet/eout:0',feed_dict={'x:0':dt[0:args.batch]})
        x = sess.run('gnet/gout:0',feed_dict={'z:0':z})

        x = (np.clip(x+0.5,0.,1.)*255.).astype('uint8')
        x = np.hstack(np.vstack(x[j] for j in range(i,100,10)) for i in range(0,10))
        cv2.imshow('img', cv2.resize(x,(1000,1000)))
        k = cv2.waitKey(0)
        if k<0 or k>255:
            break
