# -*- coding: utf-8 -*-
from __future__ import print_function
base_path = '/home/lvapeab/DATASETS/Flickr8k/'

# Inputs
split_lists = ['train_list.txt', 'val_list.txt', 'test_list.txt']
path_imgs = 'Images'
path_files = 'Annotations'

# Outputs
out_lists = ['train_list_images.txt', 'val_list_images.txt', 'test_list_images.txt']

# Code
print ('Listing all images from all data splits...')

for s, o in zip(split_lists, out_lists):
    print ('Writing on file', str(base_path + '/' + path_files + '/' + o))
    s = open(base_path + '/' + path_files + '/' + s, 'r')
    o = open(base_path + '/' + path_files + '/' + o, 'w')
    for line in s:
        line = line.strip('\n')
        this_path = path_imgs + "/" + line
        o.write(this_path + '\n')
    s.close()
    o.close()

print ('Done!')
