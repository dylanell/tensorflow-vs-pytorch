"""
Script to train a CNN or FCNN classifier with PyTorch.
"""

import argparse
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F
from util.pytorch_dataset import image_dataset
from util.data_helpers import generate_df_from_image_dataset
from model.pytorch_classifier import Classifier

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to data directory.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learn_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader threads.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train.")
    args = parser.parse_args()

    # training device - try to find a gpu, if not just use cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('[INFO]: using \'{}\' device'.format(device))

    # generate filenames/labels df from image data directory
    data_dict = generate_df_from_image_dataset(args.data_dir)

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
        data_dict['train'],
        transform=transform
    )

    # create test dataset
    test_set = image_dataset(
        data_dict['test'],
        transform=transform
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

    # initialize the model
    model = Classifier(1, 10)

    # initialize an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    # move the model to the training device
    model.to(device)

    print('[INFO]: training...')

    # train through all epochs
    for e in range(args.num_epochs):
        # loss accumulator for epoch
        epoch_loss = 0.0

        # get first batch and exit
        for i, batch in enumerate(train_loader):
            # parse batch and move to training device
            input_batch = batch['image'].to(device)
            label_batch = batch['label'].to(device)

            # compute output batch
            logits_batch = model(input_batch)

            # compute cross entropy loss (assumes raw logits as model output)
            loss = F.cross_entropy(logits_batch, label_batch)

            # add loss to loss accumulator
            epoch_loss += loss.item()

            # zero out gradient attributes for all trainabe params
            optimizer.zero_grad()

            # compute gradients w.r.t loss (repopulates gradient attribute for all trainable params)
            loss.backward()

            # update params with current gradients
            optimizer.step()

        print('[INFO]: epoch: {:d}, Loss: {:.2f}'.format(e+1, epoch_loss/i))

if __name__ == '__main__':
    main()
