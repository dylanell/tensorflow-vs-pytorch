"""
Script to train a CNN or FCNN classifier with PyTorch.
"""

import argparse
import torch
from torchvision.transforms import transforms
from util.pytorch_dataset import image_dataset
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

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to data directory.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learn_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader threads.")
    args = parser.parse_args()

    # generate filenames/labels lists from image data directory
    data_lists = generate_lists_from_image_dataset(args.data_dir)

    # define the transform chain to process each sample as it is passed to a batch
    #   1. resize the sample (image) to 32x32 (h, w)
    #   2. convert resized sample to Pytorch tensor
    #   3. normalize sample values (pixel values) using mean 0.5 and stdev 0,5; [0, 255] -> [0, 1]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # create train dataset
    train_set = image_dataset(
        data_lists['train_files'],
        data_lists['train_labels'],
        transform=transform,
    )

    # create test dataset
    test_set = image_dataset(
        data_lists['test_files'],
        data_lists['test_labels'],
        transform=transform,
    )

    # create train dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # get first batch and exit
    for i, batch in enumerate(train_loader):
        img_batch = batch['image']
        label_batch = batch['label']

        print(label_batch)
        print(img_batch.shape)
        exit()
if __name__ == '__main__':
    main()
