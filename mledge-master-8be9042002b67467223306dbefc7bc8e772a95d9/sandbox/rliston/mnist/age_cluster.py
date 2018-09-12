# git clone https://github.com/lvdmaaten/bhtsne.git
# cd bhtsne ; g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2
# CUDA_VISIBLE_DEVICES='0' python age_cluster.py
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
parser.add_argument('--images', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

with open('t10k-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    dt = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    dt = dt/255. - 0.5

with open('t10k-labels-idx1-ubyte','rb') as f:
    h = struct.unpack('>II',f.read(8))
    lt = np.fromstring(f.read(), dtype=np.uint8).astype('int32')

print 'dt.shape',dt.shape,'lt.shape',lt.shape

with tf.Session() as sess:
    with open(args.model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # run a batch of MNIST images through the encoder to generate a batch of latent vectors
    z = sess.run('enet/eout:0',feed_dict={'x:0':dt[0:args.batch]})

    if args.debug:
        for i in range(10):
            print lt[i],z[i],np.sum(np.square(z[i]))
        print 'np.mean(z,axis=0)',np.mean(z,axis=0)
        print 'np.var(z,axis=0)',np.var(z,axis=0)
        #print 'np.mean(np.square(z),axis=0)',np.mean(np.square(z),axis=0)
        print 'np.mean(z)',np.mean(z)
        print 'np.var(z)',np.var(z)

    # convert the latent vectors to tSNE format
    s=''
    for r in z:
        for c in r:
            s+=repr(c)+' '
        s+='\n'
    
    # run tSNE to project to 2 dimensions
    p = subprocess.Popen(['python','bhtsne/bhtsne.py','--no_pca','--perplexity=20.0','--verbose'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    (stdoutdata, stderrdata) = p.communicate(s)
    t = np.reshape(np.array(stdoutdata.split(),dtype='float32'),(args.batch,2),order='C')

    # plot using matplotlib
    
    if args.images:
        plt.axis('equal')
        plt.gca().set_ylim(t.min(),t.max())
        plt.gca().set_xlim(t.min(),t.max())
        for i in range(t.shape[0]):
            bb = matplotlib.transforms.Bbox.from_bounds(t[i,0],t[i,1],1,1)  
            bb2 = matplotlib.transforms.TransformedBbox(bb,plt.gca().transData)
            bbox_image = matplotlib.image.BboxImage(bb2, norm = None, origin=None, clip_on=False)
            bbox_image.set_data(cv2.cvtColor(dt[i].astype('uint8'),cv2.COLOR_GRAY2RGB))
            bbox_image.set_data(cv2.cvtColor((np.clip(dt[i]+0.5,0.,1.)*255.).astype('uint8'),cv2.COLOR_GRAY2RGB))
            plt.gca().add_artist(bbox_image)
    else:
        plt.axis('equal')
        plt.scatter(t[:,0], t[:,1],c=lt[0:args.batch],s=(matplotlib.rcParams['lines.markersize']*2) ** 2)
        cbar=plt.colorbar(values=np.arange(10))
        for i in range(10):
            cbar.ax.text(1.1,0.05+i*0.1,i)

    plt.show()
