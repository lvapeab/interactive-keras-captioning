#!/bin/bash

# Renames and splits the files from the Flickr8k dataset in order to fit the preprocessing scripts.
dataset_path=/home/lvapeab/DATASETS/Flickr30k/
token=results_20130124.token

for f in train val test; do
    echo "Processing ${f} split";
    echo -n  "" >  ${dataset_path}/Annotations/${f}_annotations.ori;
    n=0;
    for i in `cat ${dataset_path}/Annotations/${f}_list.txt | awk {'print $1'} |sort -u `
    do
        grep -Fw ${i} ${dataset_path}/Annotations/${token} >> ${dataset_path}/Annotations/${f}_annotations.ori;
        n=$(($n+1))
        if [ "$(($n % 100))" -eq "0" ] ; then
            echo "${n} samples processed.";
        fi
    done
    cat ${dataset_path}/Annotations/${f}_list.txt | awk 'BEGIN {FS="."} {print $1}' >   ${dataset_path}/Annotations/${f}_list_ids.txt
    cat ${dataset_path}/Annotations/${f}_annotations.ori | awk 'BEGIN {FS="\t"} {print $2}' > ${dataset_path}/Annotations/${f}_captions.txt
done
