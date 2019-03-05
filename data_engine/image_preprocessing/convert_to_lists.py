# -*- coding: utf-8 -*-
from __future__ import print_function
import os

import numpy as np
import tables

dataset_path = '/media/HDD_2TB/DATASETS/Flickr8k/'

features = 'InceptionV3_avg_poolnpy'  # KCNN, Scenes, Objects, VGG19
using_L2 = False
force_reload_features = False
batch_load_size = 10000
data_frame_name = 'data'

############

if using_L2:
    aux = '_L2'
else:
    aux = ''

extension = 'hdf5'
IMG_FILES = {'train': 'Annotations/train_list_ids.txt',
             'val': 'Annotations/val_list_ids.txt',
             'test': 'Annotations/test_list_ids.txt'}

IMG_FEATURES = {'train': 'Features/' + features + '/train/train_' + features + aux + '.' + extension,
                'val': 'Features/' + features + '/val/val_' + features + aux + '.' + extension,
                'test': 'Features/' + features + '/test/test_' + features + aux + '.' + extension}

CAP_FILES = {'train': 'Annotations/train_annotations.ori',
             'val': 'Annotations/val_annotations.ori',
             'test': 'Annotations/test_annotations.ori'}

# process each data split
for k in ['train', 'val', 'test']:

    print ('Processing ' + k + ' split.')
    print ('===============================')
    # Annotations/%s/train_list_ids.txt
    imgs = IMG_FILES[k]
    img_features = IMG_FEATURES[k]
    out_imgs_ids = imgs.split('.')[0] + '_ids.txt'
    out_imgs_features_store = '/'.join(img_features.split('/')[:-1])
    out_imgs_features_list = imgs.split('.')[0] + '_' + features + aux + '_features.txt'

    caps = CAP_FILES[k]
    out_caps = caps.split('.')[0] + '.txt'
    out_caps_ids = caps.split('.')[0] + '_ids.txt'

    # read img IDs
    print ('Processing images.')
    img_list_names = []
    img_list_ids = []
    img_list_ids_idx = dict()
    with open(dataset_path + imgs, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip('\n')
            line = line.split('.')[0]
            img_list_names.append(line)
            # im = line.split('_')[-1]
            img_list_ids.append(line)
            img_list_ids_idx[line] = i

    # read features
    if img_features[-3:-1] == 'np':
        feats = np.load(dataset_path + img_features).item()
        n_feat = len(feats)
        # store img features paths and .npy files
        with open(dataset_path + out_imgs_features_list, 'w') as f:
            for i, name in enumerate(img_list_names):
                im_feat_path = out_imgs_features_store + '/' + name + '.npy'
                if not os.path.isfile(dataset_path + im_feat_path) or force_reload_features:
                    np.save(dataset_path + im_feat_path, feats[name])
                f.write(im_feat_path + '\n')
                if i % 1000 == 0:
                    print ('\tStored features for %d/%d images.' % (i, n_feat))
    else:
        feat_table = tables.openFile(dataset_path + img_features, 'r')
        exec ('n_feat = feat_table.root.' + data_frame_name + '.shape[0]')
        print (n_feat, "features in the", k, "set")
        loaded_up_to = 0
        offset = 0
        # store img features paths and .npy files
        with open(dataset_path + out_imgs_features_list, 'w') as f:
            for i, name in enumerate(img_list_names):
                im_feat_path = out_imgs_features_store + '/' + name + '.npy'
                if loaded_up_to <= i or i == 0:
                    exec ('feats = feat_table.root.' + data_frame_name + '[' + str(loaded_up_to) + ':' + str(
                        min(loaded_up_to + batch_load_size, n_feat)) + ']')
                    loaded_up_to = min(loaded_up_to + batch_load_size, n_feat)
                    if i != 0:
                        offset += batch_load_size
                if not os.path.isfile(dataset_path + im_feat_path) or force_reload_features:
                    np.save(dataset_path + im_feat_path, feats[i - offset])
                f.write(im_feat_path + '\n')
                if i % 1000 == 0:
                    print ('\tStored features for %d/%d images.' % (i, n_feat))
    # store img ids
    with open(dataset_path + out_imgs_ids, 'w') as f:
        for id_name in img_list_ids:
            f.write(str(id_name) + '\n')

    if k in CAP_FILES.keys():
        print ('Processing ', k, 'captions')
        # read captions and store in imgs' order
        caps_list = [{'cap_txt': []} for i in range(len(img_list_ids))]
        caps = open(dataset_path + caps, 'r')
        i = 0
        for line in caps.readlines():
            if i % 1000 == 0:
                print ('\tProcessed %d captions.' % i)
            a = line.split('\t')
            cap_id_img = a[0].split('#')[0].split('.')[0]
            pos_cap = img_list_ids_idx[cap_id_img]
            cap_txt = a[1]
            # insert ans info in caps_list
            caps_list[pos_cap]['cap_txt'].append(cap_txt)
            i += 1
        # reformat caps in single list
        cap_single_list = []
        for a in caps_list:
            for txt in a['cap_txt']:
                cap_single_list.append(txt)
        # store caps txt
        with open(dataset_path + out_caps, 'w') as f:
            for txt in cap_single_list:
                f.write(txt.encode('utf-8'))

print ('Done')
