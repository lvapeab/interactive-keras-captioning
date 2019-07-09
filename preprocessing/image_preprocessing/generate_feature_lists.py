# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
from keras_wrapper.extra.read_write import create_dir_if_not_exists, file2list, list2file


def parse_args():
    parser = argparse.ArgumentParser("Generate feature lists.")
    parser.add_argument("-r", "--root-dir", required=False, default="./", help="Root directory of the features.")
    parser.add_argument("-fd", "--features-dir", required=False, default="Features", help="Directory (under --root-dir) containing the features.")
    parser.add_argument("-f", "--features", required=False, default="NasNetLarge", help="Name of the features.")
    parser.add_argument("-ld", "--lists-dir", required=False, default="Annotations", help="Directory (under --root-dir) containing the list splitting the dataset.")
    parser.add_argument("-l", "--list-suffix", required=False, default="_list_ids.txt", help="Suffix for the lists splitting the features. "
                                                                                             "Will be preceded with each of the options given in --splits.")
    parser.add_argument("-e", "--extension", required=False, default=".npy", help="MIME of the features.")
    parser.add_argument("-re", "--replace-extension", required=False, type=int, default=0, help="Remove this number of characters from the features names. "
                                                                                                "Set to 4 for removing MIME extensions such as '.png', '.jpg' "
                                                                                                "and replacing it by the --extension option.")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['train', 'val', 'test'], help="Splits to create.")
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help="Be verbose")
    return parser.parse_args()


def generate_feature_lists(root_dir, features_dir, features, lists_dir, list_suffix, feature_extension, replace_extension, splits, verbose):
    """

    :param root_dir: Base working directory.
    :param features_dir: Directory for storing the features.
    :param features: Features name.
    :param lists_dir: Directory (under --root-dir) containing the list splitting the dataset.
    :param list_suffix: Suffix for the lists splitting the features. Will be preceded with each of the options given in splits.
    :param feature_extension: MIME of the features.
    :param replace_extension: Remove this number of characters from the features names. Set to 4 for removing MIME extensions such as '.png', '.jpg' and replacing it by feature_extension.
    :param splits: Splits to create.
    :return:
    """
    create_dir_if_not_exists(root_dir + '/' + lists_dir + '/' + features)
    path_features = features_dir + '/' + features

    print ("Storing features in:", root_dir + '/' + lists_dir + '/' + features)
    for split in splits:
        print ('Processing split', split)
        ids = file2list(root_dir + '/' + lists_dir + '/' + split + list_suffix)
        new_ids = [path_features + '/' + split + '/' + sample_id[:-replace_extension] + feature_extension for sample_id in ids]
        list2file(root_dir + '/' + lists_dir + '/' + features + '/' + split + '_list_features.txt', new_ids)
    print ('Done!')

if __name__ == "__main__":
    args = parse_args()
    generate_feature_lists(args.root_dir, args.features_dir, args.features,
                           args.lists_dir, args.list_suffix,
                           args.extension, args.replace_extension,
                           args.splits, args.verbose)
