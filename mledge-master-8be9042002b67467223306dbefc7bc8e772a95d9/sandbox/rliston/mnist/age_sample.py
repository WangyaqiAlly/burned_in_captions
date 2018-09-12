# CUDA_VISIBLE_DEVICES='0' python age_sample.py
import argparse
import struct
import time
import numpy as np
print 'numpy ' + np.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print 'tensorflow ' + tf.__version__
import cv2
print 'cv2 ' + cv2.__version__

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=10, type=int)
parser.add_argument('--model', help='output model', default='model.proto')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

with tf.Session() as sess:
    with open(args.model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        while True:
            z = np.random.randn(100,args.m)
            z = z / np.sqrt((z*z).sum(axis=1))[:,np.newaxis]
            x = sess.run('gnet/gout:0', feed_dict={'z:0':z})
            x = (np.clip(x+0.5,0.,1.)*255.).astype('uint8')
            x = np.hstack(np.vstack(x[j] for j in range(i,100,10)) for i in range(0,10))
            cv2.imshow('img', cv2.resize(x,(1000,1000)))
            k = cv2.waitKey(0)
            if k<0 or k>255:
                break
