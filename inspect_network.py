########################################################################
#
# File:   inspect_network.py
# Author: Matt Zucker
# Date:   March 2020
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# utilities to visualize network weights and evaluate performance

import sys
import keras
import numpy as np
import cv2
import scipy.signal

import mnist_preprocess

NUM_CLASSES = 10
IMG_ROWS, IMG_COLS = 28, 28
IMG_SIZE = IMG_ROWS * IMG_COLS

######################################################################
# rescale to be in 0-255 range and make uint8

def normalize(x):
    xmin = x.min()
    xmax = x.max()
    return ((x-xmin)*(255/(xmax-xmin))).astype(np.uint8)

######################################################################
# tile small images together into one big image

def tile(images, cols, margin, bgcolor=63, zoom=1):

    irows, icols, chans = images.shape[1:]

    rows = int(np.ceil(len(images)/cols))

    h = rows*(irows + margin) + margin
    w = cols*(icols + margin) + margin

    y = margin
    x = margin
    col = 0

    vis = np.ones((h, w, chans), dtype=images.dtype) * bgcolor

    for img in images:
        vis[y:y+irows, x:x+icols] = img
        x += icols + margin
        col += 1
        if col == cols:
            x = margin
            col = 0
            y += irows + margin

    if chans == 1:
        vis = vis.reshape((h, w))

    if zoom > 1:
        vis = cv2.resize(vis, (zoom*w, zoom*h), interpolation=cv2.INTER_NEAREST)

    return vis

######################################################################
# visualize weights going into Dense layer from input image

def visualize_dense_weights(model):

    # find first dense layer
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            kernel = layer.get_weights()[0]
            break

    assert len(kernel.shape) == 2 and kernel.shape[0] == IMG_SIZE

    # reshape 784 x N array to N x 28 x 28
    # for example N = 300 if I have a 300-node hidden layer
    # we will get 1 weight image per hidden node
    kernel_images = kernel.T.reshape(-1, IMG_ROWS, IMG_COLS)

    kernel_images = normalize(kernel_images)
    
    kvis = tile(kernel_images, cols=20, margin=2)

    cv2.imshow('win', kvis)
    while cv2.waitKey(5) < 0: pass

######################################################################
# visualize weights of initial convolutional layer(s) for 1-channel
# input image

def visualize_conv_weights(model):

    display_kernel = None
    lnames = []

    for layer in model.layers:

        if isinstance(layer, keras.layers.Conv2D):

            lnames.append(layer.name)
            
            kernel = layer.get_weights()[0]

            if display_kernel is None:
                
                display_kernel = kernel
                
            else:
                
                h0, w0, num_input, num_common = display_kernel.shape
                h1, w1, num_common2, num_output = kernel.shape
                assert num_common == num_common2

                h2 = h0 + 2*(h1//2)
                w2 = w0 + 2*(w1//2)

                knew = np.zeros((h2, w2, num_input, num_output), dtype=kernel.dtype)

                for output_channel in range(num_output): # for each output channel
                    for common_channel in range(num_common): # for each common channel
                        
                        cur_common_output = kernel[:, :, common_channel, output_channel]
                        
                        for input_channel in range(num_input): # for each input channel
                            
                            prev_input_common = display_kernel[:, :, input_channel, common_channel]

                            conv = scipy.signal.correlate2d(prev_input_common, cur_common_output)
                            
                            knew[:, :, input_channel, output_channel] += conv
                            
                display_kernel = knew
                
            print('display_kernel is', display_kernel.shape, display_kernel.dtype)

            assert display_kernel.shape[2] in (1, 3)
            assert display_kernel.shape[0] == display_kernel.shape[1]

            h, w, chan, count = display_kernel.shape
            assert chan in (1, 3)

            showme = display_kernel.transpose((3, 0, 1, 2))
            assert showme.shape ==  (count, h, w, chan)
            
            showme = normalize(showme)

            kvis = tile(showme, cols=8, margin=1, zoom=8)

            print('weights for layer(s):', ', '.join(lnames))

            cv2.imshow('win', kvis)
            while cv2.waitKey(5) < 0: pass

        elif not (isinstance(layer, keras.layers.Dropout) or
                  isinstance(layer, keras.layers.InputLayer) or
                  isinstance(layer, keras.layers.Activation)):
            print('stopping visualization since I found', type(layer).__name__)
            break

def main():

    if len(sys.argv) != 2:
        print('usage: {} MODELFILE'.format(sys.argv[0]))
        sys.exit(1)

    model = keras.models.load_model(sys.argv[1])

    is_conv = False
    conv_input_shape = None

    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            is_conv = True
            conv_input_shape = layer.input_shape[1:]
            break

    activation = model.layers[-1].activation.__name__
    assert activation in ('tanh', 'softmax')
            
    model.summary()

    mnist_data = keras.datasets.mnist.load_data()
    
    (x_train, y_train), (x_test, y_test) = mnist_preprocess.process_datasets(
        mnist_data, conv=is_conv, activation=activation,
        conv_input_shape=conv_input_shape)

    if is_conv:
        visualize_conv_weights(model)
    else:
        visualize_dense_weights(model)

    loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
    print('Train loss={:7.5f}, accuracy={:7.5f}'.format(loss, accuracy))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test  loss={:7.5f}, accuracy={:7.5f}'.format(loss, accuracy))
        
if __name__ == '__main__':
    main()
