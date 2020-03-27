########################################################################
#
# File:   mnist_like_its_1998.py
# Author: Matt Zucker
# Date:   March 2020
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# this program is adapted from https://keras.io/examples/mnist_cnn/
# but more like state of the art in 1998, pre convnet, as described at
# http://yann.lecun.com/exdb/mnist/ from LeCun et al. 1998

import keras
import numpy as np

import mnist_preprocess

BATCH_SIZE = 128
NUM_HIDDEN = 300
NUM_CLASSES = 10
EPOCHS = 30

# main function
def main():

    (x_train, y_train), (x_test, y_test) = mnist_preprocess.load_datasets(
        conv=False, activation='tanh')

    model = keras.models.Sequential()

    # 784 inputs
    model.add(keras.layers.Dense(NUM_HIDDEN, activation='tanh'))

    # 300 hidden
    
    model.add(keras.layers.Dense(NUM_CLASSES, activation='tanh'))

    # 10 output

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.SGD(lr=0.05, momentum=0.05),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
    print('Train loss={:7.5f}, accuracy={:7.5f}'.format(loss, accuracy))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test  loss={:7.5f}, accuracy={:7.5f}'.format(loss, accuracy))

    filename = 'models/dense_tanh.h5'
    model.save(filename)
    print('wrote', filename)
    
if __name__ == '__main__':
    main()

