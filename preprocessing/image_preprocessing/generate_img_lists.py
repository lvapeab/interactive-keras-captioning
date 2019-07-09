# -*- coding: utf-8 -*-
from __future__ import print_function

from __future__ import print_function
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def file_exists(file):
    return os.path.isfile(file)


def parse_args():
    parser = argparse.ArgumentParser("Generate lists indexing the Padchest dataset.")
    parser.add_argument("-r", "--root-dir", required=False, default="./", help="Root directory of the dataset.")
    parser.add_argument("-i", "--image-dir", required=False, default="Images", help="Directory (under --root-dir) containing the dataset images.")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['train', 'val', 'test'], help="Splits to create.")
    parser.add_argument("-is", "--input-suffix", required=False, default="_list.txt", help="Suffix of the lists.")
    parser.add_argument("-od", "--output-dir", required=False, default='Annotations', help="Output directory.")
    parser.add_argument("-os", "--output-suffix", required=False, default='_list_images.txt', help="Output suffix for all splits.")
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help="Be verbose")
    return parser.parse_args()


def generate_lists(root_dir, image_dir, input_suffix, output_dir, output_suffix, splits, verbose):
    """
    Generate the list files required for working with an image dataset.
    :param root_dir: Root directory of the dataset.
    :param image_dir: Directory where the image files are stored (under root_dir).
    :param image_id: Name of the field in the csv containing the samples ids.
    :param labels: CSV with the dataset labels.
    :param separator: Field separator fields of output files.
    :param fields: Fields to store in the output files.
    :param splits: Splits to create. Typically, ['train', 'val', 'test'].
    :param fraction: Fractions of data to (randomly) assign to a split.
    :param output_dir: Output directory.
    :param output_suffix: Output suffix for all splits.
    :param verbose: Be verbose
    :return:
    """
    if args.verbose:
        print('Listing all images from all data splits...')

    for split in splits:
        s = split + input_suffix
        o = split + output_suffix
        if verbose:
            print('Writing on file', str(root_dir + '/' + output_dir + '/' + o))
        s = open(root_dir + '/' + output_dir + '/' + s, 'r')
        o = open(root_dir + '/' + output_dir + '/' + o, 'w')
        for line in s:
            line = line.strip('\n')
            this_path = image_dir + "/" + line
            o.write(this_path + '\n')
        s.close()
        o.close()

    print('Done!')

if __name__ == "__main__":
    args = parse_args()
    generate_lists(args.root_dir, args.image_dir, args.input_suffix, args.output_dir, args.output_suffix, args.splits, args.verbose)
