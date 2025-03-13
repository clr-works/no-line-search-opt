import numpy as np
import struct
from array import array
from os.path import join
import random

# MNIST Data Loader Class

class MnistDataloader(object):
    """
    A data loader for the MNIST dataset.
    Attributes:
    training_images_filepath (str): Path to the training images file.
    training_labels_filepath (str): Path to the training labels file.
    test_images_filepath (str): Path to the test images file.
    test_labels_filepat]h (str): Path to the test labels file.
    """

    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        """
        Reads the images and labels from the given file paths.

        Args:
        images_filepath (str): Path to the images file (either training or test).
        labels_filepath (str): Path to the labels file (either training or test).

        Returns:
        tuple: A tuple containing two lists:
        - images (list): List of images, each represented as a 2D list.
        - labels (list): List of integer labels.
        """

        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        """
        Loads training and test data.

        Returns:
        tuple: A tuple containing two tuples:
        - training data (tuple): Contains a list of training images and a list of training labels.
        - test data (tuple): Contains a list of test images and a list of test labels.
        """

        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)