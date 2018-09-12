# CUDA_VISIBLE_DEVICES='0' python age_interp.py
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
            # linear spherical interpolation between two random points : https://en.wikipedia.org/wiki/Slerp
            p0 = np.random.randn(args.m)
            p0 = p0 / np.sqrt((p0 * p0).sum())
            p1 = np.random.randn(args.m)
            p1 = p1 / np.sqrt((p1 * p1).sum())
            z = np.empty((100,10),dtype='float32')
            a = np.arccos(np.dot(p0,p1))
            for i in range(0,100):
                t = i*0.01
                z[i] = (np.sin((1.-t)*a)/np.sin(a))*p0 + (np.sin(t*a)/np.sin(a))*p1

            x = sess.run('gnet/gout:0', feed_dict={'z:0':z})
            x = (np.clip(x+0.5,0.,1.)*255.).astype('uint8')
            x = np.hstack(np.vstack(x[j] for j in range(i,100,10)) for i in range(0,10))
            cv2.imshow('img', cv2.resize(x,(1000,1000)))
            k = cv2.waitKey(0)
            if k<0 or k>255:
                break
