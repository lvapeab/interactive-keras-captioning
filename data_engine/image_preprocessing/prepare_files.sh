#!/usr/bin/env bash

# Renames and splits the files from the Flickr8k dataset in order to fit the preprocessing scripts.

dataset_path=/home/lvapeab/DATASETS/Flickr8k/

token=results_20130124.token

for f in train val test; do
    echo "Processing ${f} split"
    echo -n  "" >  ${dataset_path}/Annotations/${f}_annotations.ori
    for i in `cat ${dataset_path}/Annotations/${f}_list.txt | awk {'print $1'}`
    do
        grep ${i} ${dataset_path}/Annotations/${token} >> ${dataset_path}/Annotations/${f}_annotations.ori
    done
    cat ${dataset_path}/Annotations/${f}_list.txt | awk 'BEGIN {FS="."} {print $1}' >   ${dataset_path}/Annotations/${f}_list_ids.txt
done
