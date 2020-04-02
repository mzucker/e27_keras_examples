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

def pad(x, input_shape):

    n = len(x)
    assert x.shape == (n, IMG_ROWS, IMG_COLS, 1)

    irows, icols, ichan = input_shape

    assert irows >= IMG_ROWS
    assert icols >= IMG_COLS
    assert ichan >= 1

    result = np.zeros((n,) + input_shape, dtype=x.dtype)

    row = (irows - IMG_ROWS)//2
    col = (icols - IMG_COLS)//2

    result[:, row:row+IMG_ROWS, col:col+IMG_COLS, :] = x

    return result

# convert from stack of images to stack of row vectors
def process_images(x, conv, conv_input_shape):

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

        if input_shape is not None:
            x = pad(x, conv_input_shape)

    elif input_shape is not None:
        
        raise RuntimeError('input_shape is only for conv. nets')
    
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


def process_dataset(img, lbl, conv, activation, conv_input_shape=None):

    assert activation in ('tanh', 'softmax')

    x = process_images(img, conv, conv_input_shape)

    if activation == 'tanh':
        y = posneg1_from_labels(lbl)
    else:
        y = keras.utils.to_categorical(lbl, NUM_CLASSES)

    return x, y

def process_datasets(datasets, conv, activation, conv_input_shape=None):
    
    outputs = []
    
    for (img, lbl) in datasets:
        outputs.append(process_dataset(img, lbl,
                                       conv=conv,
                                       activation=activation,
                                       conv_input_shape=conv_input_shape))
        
    return tuple(outputs)
    
    
def load_datasets(conv, activation, conv_input_shape=None):
    datasets = keras.datasets.mnist.load_data()
    return process_datasets(datasets,
                            conv=conv,
                            activation=activation,
                            conv_input_shape=conv_input_shape)

    
