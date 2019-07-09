#!/bin/bash

# Renames and splits the files from the Flickr8k dataset in order to fit the preprocessing scripts.
dataset_path=$1
token=$2

for f in train val test; do
    echo "Processing ${f} split";
    echo -n  "" >  ${dataset_path}/${f}_annotations.ori;
    n=0;
    for i in `cat ${dataset_path}/${f}_list.txt | awk {'print $1'} |sort -u `
    do
        grep -Fw ${i} ${dataset_path}/${token} >> ${dataset_path}/${f}_annotations.ori;
        n=$(($n+1))
        if [ "$(($n % 100))" -eq "0" ] ; then
            echo "${n} samples processed.";
        fi
    done
    cat ${dataset_path}/${f}_list.txt | awk 'BEGIN {FS="."} {print $1}' >   ${dataset_path}/${f}_list_ids.txt
    cat ${dataset_path}/${f}_annotations.ori | awk 'BEGIN {FS="\t"} {print $2}' > ${dataset_path}/${f}_captions.txt
    rm ${dataset_path}/${f}_annotations.ori
done
