"""
Pytorch dataset classes.
Reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import torch
from PIL import Image

class image_dataset(torch.utils.data.Dataset):
    """
    Make a PyTorch dataset from a directory of images and a labels csv.
    """

    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # read image and get label
        # NOTE: image must be PIL image for standard PyTorch transforms
        image = Image.open(self.image_files[idx])
        label = self.labels[idx]

        # apply any image transform
        if self.transform:
            image = self.transform(image)

        # construct packaged sample
        sample = {'image': image, 'label': label}

        return sample

class raw_dataset(torch.utils.data.Dataset):
    """
    Make a Pytorch dataset from provided samples and labels.
    """

    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # get sample and label by idx
        pack = {0: self.samples[idx], 1: self.labels[idx]}

        # add transform
        if self.transform:
            pack = self.transform(pack)

        return pack
