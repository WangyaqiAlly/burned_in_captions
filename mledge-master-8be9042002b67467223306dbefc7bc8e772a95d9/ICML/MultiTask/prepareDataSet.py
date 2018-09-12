import numpy as np
import random
from shutil import copy2
import os
import sys

celeb    = '/home2/dataset/Celeb-A'

srcdir               = os.path.join(celeb, 'img_align_celeba')
train_unlabelled_dir = os.path.join(celeb, 'train-unlabelled')
train_labelled_dir   = os.path.join(celeb, 'train-labelled')
transmitted_dir      = os.path.join(celeb, 'transmitted')
testdir              = os.path.join(celeb, 'test')
nfiles = 202599

frac_data  = 0.9
frac_label = 0.1
frac_transmit = [0.5, 1.0, 2.0, 5.0, 10.0]

thr_test     = int(frac_data*nfiles)
thr_transmit = [int(x*frac_data*nfiles/100.0) for x in frac_transmit]
thr_labelled = int(frac_data*frac_label*nfiles)

for d in (train_unlabelled_dir, train_labelled_dir, testdir):
    if not os.path.exists(d):
        os.makedirs(d)

for x in frac_transmit:
    p = os.path.join(celeb, 'transmit-' + str(x))
    if not os.path.exists(p):
        os.makedirs(p)
        
a = range(nfiles)
random.shuffle(a)

for k,i in enumerate(a):
    fname = '{0:06}.jpg'.format(i+1)
    if k > thr_test:
        dst = testdir
    else:
        dst = train_unlabelled_dir
    fpath = os.path.join(srcdir, fname)
    copy2(fpath, dst)

    if k < thr_labelled:
        copy2(fpath, train_labelled_dir)

    for j,p in enumerate(frac_transmit):
        if k < thr_transmit[j]:
            d = os.path.join(celeb, 'transmit-' + str(p))
            copy2(fpath, d)

def distribute_sites(dname, n):
    dir_dict = dict()
    for s in range(n):
        d = dname + '-site-' + str(s)
        dir_dict[s] = d
        if not os.path.exists(d):
            os.makedirs(d)
    files = os.listdir(dname)
    for i, f in enumerate(files):
        dst = dir_dict[i%n]
        copy2(os.path.join(dname, f), dst)

nsites=2
distribute_sites(train_unlabelled_dir, nsites)
distribute_sites(train_labelled_dir,   nsites)
