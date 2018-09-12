#!/bin/bash

dest_central_train=./node_central/init/train.hdf
dest_central_test=./node_central/init/test.hdf
init_record=./record/init/retina_record_fold_1.hdf
fold1=./data/fold_1.hdf
testdata=./data/test.hdf
mkdir ./node_central/init
cp $fold1 $dest_central_train
cp $testdata $dest_central_test
CUDA_VISIBLE_DEVICES=0 python test.py --name=node_central --batch_num=128 --epoch=200 --record_name=init --mode=0

