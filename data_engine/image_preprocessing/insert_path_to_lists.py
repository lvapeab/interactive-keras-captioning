# -*- coding: utf-8 -*-
from __future__ import print_function
dataset_path = '/media/HDD_2TB/DATASETS/Flickr8k/'
prefix_path = '/media/HDD_2TB/DATASETS/Flickr30k/Images/'  # prefix path not included in IMG_FILES but must be included in OUT_IMG_FILES

prefix_specific = {'train': '', 'val': '', 'test': ''}

IMG_FILES = {'train': 'Annotations/train_list.txt',
             'val': 'Annotations/val_list.txt',
             'test': 'Annotations/test_list.txt'}

OUT_IMG_FILES = {'train': 'Annotations/train_list_images.txt',
                 'val': 'Annotations/val_list_images.txt',
                 'test': 'Annotations/test_list_images.txt'}

# Process each data split
for s in IMG_FILES.keys():
    print ("Processing set " + s)
    out_f = open(dataset_path + '/' + OUT_IMG_FILES[s], 'w')
    with open(dataset_path + '/' + IMG_FILES[s]) as in_f:
        for line in in_f:
            line = line.rstrip('\n')
            out_f.write(prefix_path + '/' + prefix_specific[s] + '/' + line + '\n')
print ("Done")
