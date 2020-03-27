########################################################################
#
# File:   cnn_keras_tutorial.py
# Author: Matt Zucker
# Date:   March 2020
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# fairly faithfully follows tutorial at https://keras.io/examples/mnist_cnn/

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

    # input is 28 x 28 x 1

    # kernel shape is (32, 3, 3, 1)
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                                  activation='relu',
                                  input_shape=input_shape))

    # shape is now 26 x 26 x 32

    # kernel shape is (64, 3, 3, 32)
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

    # shape is now 24 x 24 x 64

    # apply a maximum operation over 2x2 neighborhoods in the image
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # shape is now 12 x 12 x 64
    
    model.add(keras.layers.Dropout(0.25))

    # dropout doesn't change shape

    model.add(keras.layers.Flatten())

    # shape is now flat array of 9,216 = (12*12*64)
    
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

    filename = 'models/cnn_keras_tutorial.h5'

    model.save(filename)
    print('wrote', filename) 
    
if __name__ == '__main__':
    main()

