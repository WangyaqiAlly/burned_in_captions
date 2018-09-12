#!/bin/bash

model_name=baseline
random_rate=1.00
GPU_NUM=3
EPOCH_NUM=200
BATCH_NUM=128
mode=1


if ! [ -d ./node_central/$model_name ]; then
    mkdir ./node_central/$model_name;
fi

if ! [ -d ./node_edge/$model_name ]; then
    mkdir ./node_edge/$model_name;
fi

rm -r ./node_central/$model_name/*
rm -r ./node_edge/$model_name/*
rm -r ./record/$model_name/*

fold_name=( fold_1 fold_2 fold_3 fold_4 fold_5 fold_6 fold_7 fold_8 fold_9)

source_path=./data/${fold_name[0]}.hdf
dest_path=./node_central/$model_name/train.hdf
echo moving from $source_path $dest_path;
cp $source_path $dest_path;

source_path=./data/test.hdf
dest_path=./node_central/$model_name/test.hdf
echo moving from $source_path $dest_path;
cp $source_path $dest_path;

for i in 1 2 3 4 5 6 7 8
do
    source_path=./data/${fold_name[i]}.hdf
    dest_path=./node_edge/$model_name/
    echo moving from $source_path $dest_path;
    cp $source_path $dest_path;
done

source_path=./data/test.hdf
dest_path=./node_edge/$model_name/
echo moving from $source_path $dest_path;
cp $source_path $dest_path;

CUDA_VISIBLE_DEVICES=$GPU_NUM python test.py --name=node_central --batch_num=$BATCH_NUM --epoch=$EPOCH_NUM --record_name=$model_name --mode=$mode --rate=$random_rate
init_record=./record/init/retina_record_fold_1*
cp $init_record ./record/$model_name/

