########################################################################
#
# File:   dense_softmax_relu.py
# Author: Matt Zucker
# Date:   March 2020
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# changes the 784->300->10 dense network from mnist_like_its_1998.py:
#
#  * change hidden layer activation tanh -> relu
#  * change output layer activation tanh -> softmax 
#  * compare dropout vs no dropout

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
        conv=False, activation='softmax')
    
    for do_dropout in (False, True):

        model = keras.models.Sequential()

        if do_dropout:
            model.add(keras.layers.Dropout(rate=0.15))

        model.add(keras.layers.Dense(NUM_HIDDEN, activation='relu'))

        if do_dropout:
            model.add(keras.layers.Dropout(rate=0.5))

        model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))

        # cross-entropy is multiclass analog to logistic regression
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
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

        if do_dropout:
            filename = 'models/dense_softmax_relu_with_dropout.h5'
        else:
            filename = 'models/dense_softmax_relu_no_dropout.h5'
            
        model.save(filename)
        print('wrote', filename)
    
if __name__ == '__main__':
    main()

