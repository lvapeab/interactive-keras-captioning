#!/usr/bin/env bash

# Renames and splits the files from the Flickr8k dataset in order to fit the preprocessing scripts.

nrefs=5
dataset_path=/media/HDD_2TB/DATASETS/MSCOCO/Annotations/

for f in val test; do
    echo "Processing ${f} split"
    for n in `seq 0 $(( ${nrefs} - 1 ))`; do
        grep "#${n}" ${dataset_path}/${f}_annotations.ori | awk 'BEGIN{FS="\t"}{print $2}' >  ${dataset_path}/${f}_${n}.refs
    done
done
