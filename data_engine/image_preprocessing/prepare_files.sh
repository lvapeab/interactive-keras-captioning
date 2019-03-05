#!/usr/bin/env bash

# Renames and splits the files from the Flickr8k dataset in order to fit the preprocessing scripts.

dataset_path=/data/DATASETS/Flickr30k/
#dataset_path=/media/HDD_2TB/DATASETS/Flickr8k/

token=results_20130124.token

#mv ${dataset_path}/Annotations/Flickr_8k.trainImages.txt ${dataset_path}/Annotations/train_list.txt 2> /dev/null
#mv ${dataset_path}/Annotations/Flickr_8k.devImages.txt ${dataset_path}/Annotations/val_list.txt 2> /dev/null
#mv ${dataset_path}/Annotations/Flickr_8k.testImages.txt ${dataset_path}/Annotations/test_list.txt 2> /dev/null

for f in train val test; do
    echo "Processing ${f} split"
    echo -n  "" >  ${dataset_path}/Annotations/${f}_annotations.ori
    for i in `cat ${dataset_path}/Annotations/${f}_list.txt | awk {'print $1'}`
    do
        grep ${i} ${dataset_path}/Annotations/${token} >> ${dataset_path}/Annotations/${f}_annotations.ori
    done
done
