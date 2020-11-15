"""
Various Python functions for processing data.
"""

import pandas as pd


def generate_df_from_image_dataset(path):
    # read train/test label files to dataframe
    train_df = pd.read_csv('{}train_labels.csv'.format(path))
    test_df = pd.read_csv('{}test_labels.csv'.format(path))

    # convert filename column to absolute paths
    train_df['Filename'] = train_df['Filename'] \
        .map(lambda x: '{}train/{}'.format(path, x))
    test_df['Filename'] = test_df['Filename'] \
        .map(lambda x: '{}test/{}'.format(path, x)).to_list()

    # package data to dictionary
    data_dict = {'train': train_df, 'test': test_df}

    return data_dict
