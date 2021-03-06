"""
Script to train a CNN classifier with TensorFlow.
"""

import time
import yaml
import tensorflow as tf

from util.data_helpers import generate_df_from_image_dataset
from util.tensorflow_helpers import ImageDatasetBuilderVanilla
from model.tensorflow_classifier import Classifier

AUTOTUNE = tf.data.experimental.AUTOTUNE


def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('[INFO]: found {} GPUs:'.format(len(gpus)))

    # generate filenames/labels df from image data directory
    data_dict = generate_df_from_image_dataset(
        config['dataset_directory']
    )

    # get number of classes in labels
    num_class = data_dict['train']['Label'].nunique()

    # initialize the dataset builder
    dataset_builder = ImageDatasetBuilderVanilla(
        image_size=config['input_dimensions'][:-1],
        batch_size=config['batch_size']
    )

    # create training/testing datasets
    train_ds = dataset_builder.build(data_dict['train'])
    test_ds = dataset_builder.build(data_dict['test'])

    # initialize model
    model = Classifier(config['input_dimensions'], num_class)

    # create loss object
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )

    # create optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate']
    )

    # create metrics that accumulate over epoch
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'
    )
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy'
    )

    print('[INFO]: training...')

    for e in range(config['number_epochs']):
        # get epoch start time
        epoch_start = time.time()

        # reset metrics at the start of each epoch
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        # run through training epoch
        for batch_images, batch_labels in train_ds:
            # open a gradient tape scope to 'watch' operations
            with tf.GradientTape() as tape:
                # compute predictions
                batch_logits = model(batch_images)

                # compute loss
                batch_loss = loss_object(batch_labels, batch_logits)

            # compute loss gradients w.r.t model params
            gradient = tape.gradient(
                batch_loss,
                model.trainable_variables
            )

            # update params with gradients
            optimizer.apply_gradients(
                zip(gradient, model.trainable_variables)
            )

            # add loss to train loss accumulator object
            train_loss(batch_loss)

            # add accuracy to train accuracy accumulator
            train_acc(batch_labels, batch_logits)

        # run through training epoch
        for batch_images, batch_labels in test_ds:
            # compute predictions
            batch_logits = model(batch_images)

            # compute loss
            batch_loss = loss_object(batch_labels, batch_logits)

            # add loss to test loss accumulator object
            test_loss(batch_loss)

            # add accuracy to test accuracy accumulator
            test_acc(batch_labels, batch_logits)

        # compute epoch time
        epoch_time = time.time() - epoch_start

        # print epoch metrics
        template = '[INFO]: Epoch {}, Epoch Time {:.2f}s, Train Loss: ' \
                   '{:.2f}, Train Accuracy: {:.2f}, Test Loss: {:.2f}, ' \
                   'Test Accuracy: {:.2f}'
        print(template.format(e + 1, epoch_time, train_loss.result(),
                              100 * train_acc.result(), test_loss.result(),
                              100 * test_acc.result()))


if __name__ == '__main__':
    main()
