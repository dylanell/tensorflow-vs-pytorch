"""
CNN classifier implemented in PyTorch.
"""

import torch
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    """

    """

    # initialize and define all layers within the model
    def __init__(self, in_channels, out_dimension):
        # run base class initializer
        super(Classifier, self).__init__()

        # define convolution layers
        self.conv_1 = torch.nn.Conv2d(in_channels, 32, 3, stride=1, padding=1)  # in_dim / 1
        self.conv_2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)           # in_dim / 2
        self.conv_3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)          # in_dim / 2
        self.conv_4 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)         # in_dim / 2

        # define fully connected layers
        self.fc_1 = torch.nn.Linear(256 * 4 * 4, out_dimension)

    # compute forward propagation of input x using model layers (dynamic build)
    def forward(self, x):
        # compute output
        z_1 = F.relu(self.conv_1(x))                                        # 32x32x32
        z_2 = F.relu(self.conv_2(z_1))                                      # 64x16x16
        z_3 = F.relu(self.conv_3(z_2))                                      # 128x8x8
        z_4 = F.relu(self.conv_4(z_3))                                      # 256x4x4
        z_4_flat = torch.flatten(z_4, start_dim=1)                          # 4096
        z_5 = self.fc_1(z_4_flat)                                           # 10

        return z_5
