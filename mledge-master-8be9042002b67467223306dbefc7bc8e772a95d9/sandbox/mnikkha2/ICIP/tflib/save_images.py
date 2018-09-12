"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave
import cv2

def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    rowImg = []
    for k in range(0,X.shape[0],nh):
        rowImg.append(np.hstack(X[j] for j in range(k,k+nw)))
    xgen = np.vstack(rowImg)
    #xgen = np.vstack(X[j:j+nw] for j in range(0,X.shape[0],nw))
    #print xgen.shape
    #theImage = cv2.resize(xgen,(1000,1000))
    #cv2.imshow('img', xgen)
    #k = cv2.waitKey(0)
    #if k==1114083: # ctrl-c to exit
    #break
    cv2.imwrite(save_path, xgen)
    #for n, x in enumerate(X):
    #    j = n/nw
    #    i = n%nw
    #    img[j*h:j*h+h, i*w:i*w+w] = x

    #imsave(save_path, img)
