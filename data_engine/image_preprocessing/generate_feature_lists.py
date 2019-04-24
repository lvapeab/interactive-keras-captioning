# -*- coding: utf-8 -*-
from __future__ import print_function
from keras_wrapper.extra.read_write import create_dir_if_not_exists, file2list, list2file
base_path = '/home/lvapeab/DATASETS/Flickr30k/'

# Inputs
path_lists = 'Annotations'
features_name = 'InceptionV3'
path_features = 'Features' + '/' + features_name
feature_extension = '.npy'

ID_LISTS  = {'train': 'train_list_ids.txt',
             'val': 'val_list_ids.txt',
             'test': 'test_list_ids.txt'}

# Outputs
OUT_FEATURES_LISTS = {'train': features_name + '/train_list_features.txt',
                      'val': features_name + '/val_list_features.txt',
                      'test':  features_name + '/test_list_features.txt'
                      }

create_dir_if_not_exists(base_path + '/' + path_lists + '/' + features_name)

for split in ['train', 'val', 'test']:
    print ('Processing split', split)
    ids = file2list(base_path + '/' + path_lists + '/' + ID_LISTS[split])
    new_ids = [path_features + '/' + sample_id + feature_extension for sample_id in ids]
    list2file(base_path + '/' + path_lists + '/' + OUT_FEATURES_LISTS[split], new_ids)
print ('Done!')
