########################################################################
#
# File:   mnist_preprocess.py
# Author: Matt Zucker
# Date:   March 2020
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# utilities to load and preprocess MNIST data for keras demos

import numpy as np
import keras

# input image dimensions
IMG_ROWS, IMG_COLS = 28, 28

NUM_CLASSES = 10

# convert from stack of images to stack of row vectors
def process_images(x, conv):

    # reshape 3D array -> 2D array
    assert x.shape[1:] == (IMG_ROWS, IMG_COLS)
    x = x.reshape(-1, IMG_ROWS*IMG_COLS).astype(np.float32)

    # subtract off row-wise means
    x -= x.mean(axis=1).reshape(-1, 1)

    # divide off row-wise standard deviations
    x /= x.std(axis=1).reshape(-1, 1)

    if conv:

        if keras.backend.image_data_format() == 'channels_first':
            input_shape = (1, IMG_ROWS, IMG_COLS)
        else:
            input_shape = (IMG_ROWS, IMG_COLS, 1)

        x = x.reshape((-1,) + input_shape)
    
    return x

# convert from numerical labels to matrix of +/- 1's
def posneg1_from_labels(labels):

    num_samples = len(labels)

    # create big matrix of -1's
    y = -np.ones((num_samples, NUM_CLASSES), dtype=np.float32)

    # one index per row
    idx = np.arange(num_samples)

    # for each row, set the column given by labels to +1
    y[idx, labels] = 1

    return y


def process_dataset(img, lbl, conv, activation):

    assert activation in ('tanh', 'softmax')

    x = process_images(img, conv)

    if activation == 'tanh':
        y = posneg1_from_labels(lbl)
    else:
        y = keras.utils.to_categorical(lbl, NUM_CLASSES)

    return x, y

def process_datasets(datasets, conv, activation):
    
    outputs = []
    
    for (img, lbl) in datasets:
        outputs.append(process_dataset(img, lbl, conv, activation))
        
    return tuple(outputs)
    
    
def load_datasets(conv, activation):
    datasets = keras.datasets.mnist.load_data()
    return process_datasets(datasets, conv, activation)

    
