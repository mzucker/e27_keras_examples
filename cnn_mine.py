########################################################################
#
# File:   cnn_mine.py
# Author: Matt Zucker
# Date:   March 2020
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# attempt to improve on keras tutorial with deeper network that has
# 80% fewer parameters

import keras
import numpy as np

import mnist_preprocess

BATCH_SIZE = 128
NUM_HIDDEN = 300
NUM_CLASSES = 10
EPOCHS = 12

def main():
    
    (x_train, y_train), (x_test, y_test) = mnist_preprocess.load_datasets(
        conv=True, activation='softmax')

    input_shape = x_train.shape[1:]
    print('input_shape =', input_shape)

    model = keras.models.Sequential()

    shapes = [(16, 32), (64,)]

    for shape_tuple in shapes:

        for channels in shape_tuple:

            # kernel shape is (32, 3, 3, 1)
            model.add(keras.layers.Conv2D(channels, kernel_size=(3, 3),
                                          activation='relu',
                                          input_shape=input_shape))

            input_shape = model.layers[-1].output_shape[1:]
            print('after conv2d now input shape is', input_shape)

        # apply a maximum operation over 2x2 neighborhoods in the image
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.15))
        
        input_shape = model.layers[-1].output_shape[1:]
        print('after max pooling now input shape is', input_shape)

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128, activation='relu'))
    
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])


    for layer in model.layers:
        print('layer {} has input_shape {}, output_shape {}'.format(
            layer.name, layer.input_shape, layer.output_shape))
    
    trainable_count =  keras.utils.layer_utils.count_params(model.trainable_weights)
    print('total trainable parameters:', trainable_count)
    
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
    print('Train loss={:7.5f}, accuracy={:7.5f}'.format(loss, accuracy))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test  loss={:7.5f}, accuracy={:7.5f}'.format(loss, accuracy))

    filename = 'models/cnn_mz.h5'

    model.save(filename)
    print('wrote', filename) 
    
if __name__ == '__main__':
    main()

