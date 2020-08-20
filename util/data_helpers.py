"""
Various functions to help with processing data.
"""

import pandas as pd

def generate_lists_from_image_dataset(path):
    # read train/test label files
    train_labels_df = pd.read_csv('{}train_labels.csv'.format(path))
    test_labels_df = pd.read_csv('{}test_labels.csv'.format(path))

    # convert labels column to list
    train_labels = train_labels_df['Label'].to_list()
    test_labels = test_labels_df['Label'].to_list()

    # convert filename column to list of absolute paths
    train_files = train_labels_df['Image Filename'].map(lambda x: \
        '{}train/{}'.format(path, x)).to_list()
    test_files = test_labels_df['Image Filename'].map(lambda x: \
        '{}test/{}'.format(path, x)).to_list()

    # package data to dictionary
    data_lists = {
        'train_files': train_files,
        'train_labels': train_labels,
        'test_files': test_files,
        'test_labels': test_labels,
    }

    return data_lists
