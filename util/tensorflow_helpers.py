"""
TensorFlow helper classes and functions.
"""

import tensorflow as tf

# for autotuning multi-threading/parallelization parameters
AUTOTUNE = tf.data.experimental.AUTOTUNE


class ImageDatasetBuilderVanilla():
    """
    Given a dataframe of the form [img_paths, labels], construct a TensorFlow
    dataset object and perform all of the standard image dataset processing
    functions (resizing, standardization, etc.).
    """

    def __init__(self, image_size=(32, 32), batch_size=64):
        # set global attributes
        self.image_size = image_size
        self.batch_size = batch_size

    def build(self, dataframe):
        # create tf dataset from (file. label) tensor slices
        dataset = tf.data.Dataset.from_tensor_slices((
            dataframe['Filename'].tolist(),
            dataframe['Label'].tolist()
        ))

        # map the image processing function to process image paths
        dataset = dataset.map(
            self.__read_and_process_image,
            num_parallel_calls=AUTOTUNE
        )

        # set dataset to randomly shuffle order
        dataset = dataset.shuffle(buffer_size=1000)

        # set dataset to return batches of multiple elements
        dataset = dataset.batch(self.batch_size)

        # set dataset to prefetch elements for better feed performance
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset

    def __read_and_process_image(self, filename, label):
        # read image file
        image_file = tf.io.read_file(filename)

        # decode image to tensor
        # expand_animations=False is needed to get image with 'shape'
        # reference: https://stackoverflow.com/a/59944421
        image = tf.io.decode_image(image_file, expand_animations=False)

        # convert image dtype tp float32
        image = tf.image.convert_image_dtype(image, tf.float32)

        # resize image (32 can be global attribute when using this in a class)
        image = tf.image.resize(
            image,
            [self.image_size[0], self.image_size[1]]
        )

        # standardize image to (mean=0, stdev=1)
        image = tf.image.per_image_standardization(image)

        return image, label
