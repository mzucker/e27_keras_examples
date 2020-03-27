########################################################################
#
# File:   keras_xor.py
# Author: Matt Zucker
# Date:   February 2020
#
# Written for ENGR 27 - Computer Vision
#
########################################################################

# shows how to do tiny 2-layer nnet to solve XOR problem with keras

import keras
import numpy as np

def main():

    # XOR input data
    x = np.array([[-1, -1],
                  [-1,  1],
                  [ 1, -1],
                  [ 1,  1]], dtype=np.float32)

    # XOR output data
    y = np.array([[-1],
                  [ 1],
                  [ 1],
                  [-1]], dtype=np.float32)

    # create a sequential model
    model = keras.models.Sequential()

    # for this particular dataset I found it helpful to put in a
    # bias_initializer for each layer but you don't usually need to do
    # that.
    model.add(keras.layers.Dense(2, activation='tanh',
                                 bias_initializer='random_normal'))
    
    model.add(keras.layers.Dense(1, activation='tanh',
                                 bias_initializer='random_normal'))

    # compile the model.
    #
    # you can use a GIANT learning rate of 0.5 with this trivial dataset
    # probably you would usually use a much smaller one (default is 0.01)
    
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.SGD(lr=0.5),
                  metrics=['accuracy'])

    # fit the data
    model.fit(x, y, batch_size=4, epochs=200, verbose=1)

    # show output for this dataset
    output = model.predict(x)

    print('x:\n', x, sep='')
    print('desired output:', y.flatten())
    print('actual output: ', output.flatten())
    print('error:         ', np.mean((y-output)**2))

if __name__ == '__main__':
    main()
    
