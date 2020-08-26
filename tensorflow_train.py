"""
Script to train a CNN or FCNN classifier with TensorFlow.
"""

import argparse
import tensorflow as tf
from util.data_helpers import generate_df_from_image_dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

def dataset_map_func(filename, label):
    # read image file
    image_file = tf.io.read_file(filename)

    # TODO do image preprocessing here

    # decode image to tensor
    image = tf.io.decode_image(image_file)

    return image, label

def configure_dataset(dataset, batch_size):
    # set dataset to cache files for more efficient retrieval
    dataset = dataset.cache()

    # set dataset to randomly shuffle order
    dataset = dataset.shuffle(buffer_size=1000)

    # set dataset to return batches of multiple elements
    dataset = dataset.batch(batch_size)

    # set dataset to prefetch elements for better feed performance
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to data directory.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learn_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader threads.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train.")
    args = parser.parse_args()

    # generate filenames/labels df from image data directory
    data_dict = generate_df_from_image_dataset(args.data_dir)

    # create train set from (file, label) tensor slices
    train_set = tf.data.Dataset.from_tensor_slices((
        data_dict['train']['Filename'].tolist(),
        data_dict['train']['Label'].tolist()
    ))

    # map train set to process images and labels
    train_set = train_set.map(
        dataset_map_func,
        num_parallel_calls=AUTOTUNE
    )

    # configure train set for performance
    train_set = configure_dataset(train_set, args.batch_size)

    # create test set from (file, label) tensor slices
    test_set = tf.data.Dataset.from_tensor_slices((
        data_dict['test']['Filename'].tolist(),
        data_dict['test']['Label'].tolist()
    ))

    # map test set to process images and labels
    test_set = test_set.map(
        dataset_map_func,
        num_parallel_calls=AUTOTUNE
    )

    # configure train set for performance
    test_set = configure_dataset(test_set, args.batch_size)

    print(train_set)
    print(test_set)

if __name__ == '__main__':
    main()
