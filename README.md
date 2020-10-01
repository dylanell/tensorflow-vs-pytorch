# tensorflow-vs-pytorch
A comparison of TensorFlow vs. PyTorch with classification on MNIST.

### Environment:

1. Python 3.7.4

### Python Packages:

1. torch
2. torchvision
3. tensorflow
4. pandas
5. PIL

### Image Dataset Format:

This project assumes you have the MNIST dataset pre-configured locally on your machine in the format described below. My [dataset-helpers](https://github.com/dylanell/dataset-helpers) Github project also contains tools that perform this local configuration automatically within the `mnist` directory.

The MNIST dataset consists of images of written numbers (0-9) with corresponding labels. The dataset can be accessed a number of ways using Python packages (`mnist`, `torchvision`, `tensorflow_datasets`, etc.), or it can be downloaded directly from the [MNIST homepage](http://yann.lecun.com/exdb/mnist/). In order to demonstrate an image-based data pipeline in a standard way and demonstrate how to use memory-efficient dataloaders in both TensorFlow and Pytorch, we organize the MNIST dataset into training/testing directories of raw image files (`png` or `jpg`) accompanied by a `csv` file listing one-to-one correspondences between the image file names and their label. In general, this "generic image dataset format" is summarized by the directory tree structure below.

```
dataset_directory/
|__ train_labels.csv
|__ test_labels.csv
|__ train/
|   |__ train_image_01.png
|   |__ train_image_02.png
|   |__ ...
|__ test/
|   |__ test_image_01.png
|   |__ test_image_02.png
|   |__ ...   
```

Each labels `csv` file has the format:

```
Filename, Label
train_image_01.png, 4
train_image_02.png, 7
...
```

If you would like to re-use the code here to work with other image datasets, just format any new image dataset to follow the outline above and be sure to edit corresponding hyperparameters in the `config.yaml` file.

### Training

Training hyperparameters are pulled from the `config.yaml` configuration file and can be changed by editing the file contents.

Train the TensorFlow classifier by running:

```
$ python tensorflow_train.py
```

Train the PyTorch classifier by running:

```
$ python pytorch_train.py </path/to/dataset/directory>
```

### References:

1. PyTorch Dataset Pipelines:
  * https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
2. TensorFlow Training:
  * https://www.tensorflow.org/tutorials/quickstart/advanced
3. TensorFlow Dataset Pipelines:
  * https://www.tensorflow.org/api_docs/python/tf/data/Dataset
  * https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control
4. TensorFlow GPU Support:
  * https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d
