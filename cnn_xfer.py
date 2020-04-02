########################################################################
#
# File:   cnn_xfer.py
# Author: Matt Zucker
# Date:   April 2020
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# attempt to demonstrate transfer learning on MNIST by using pre-trained
# CIFAR 10 model

import sys
import keras
import numpy as np

import mnist_preprocess

BATCH_SIZE = 128
NUM_HIDDEN = 300
NUM_CLASSES = 10

ADAPT_EPOCHS = 2
TUNE_EPOCHS = 10


def main():

    orig_model = keras.models.load_model('models/keras_cifar10_trained_model.h5')

    conv_input_shape = orig_model.layers[0].input_shape[1:]

    print('conv_input_shape=', conv_input_shape)

    (x_train, y_train), (x_test, y_test) = mnist_preprocess.load_datasets(
        conv=True, activation='softmax',
        conv_input_shape=conv_input_shape)

    flatten_layer = None

    for layer in orig_model.layers:
        # freezing the weights of the original model
        # weights won't be updated during first round of training 
        layer.trainable = False
        if isinstance(layer, keras.layers.Flatten):
            flatten_layer = layer

    if flatten_layer is None:
        print('no flatten layer :(')
        sys.exit(1)

    print('*** OLD MODEL ***')
    orig_model.summary()

    # now make a new model using keras functional API

    dense0 = keras.layers.Dense(128, activation='relu')
    dropout = keras.layers.Dropout(0.5, name='dense_dropout')
    dense1 = keras.layers.Dense(NUM_CLASSES, activation='softmax')

    # input to new model is same as input to old model
    inp = orig_model.input

    # take the output from the flatten layer
    # and send it to the first new dense layer
    out = dense0(flatten_layer.output)

    # send it to the the dropout layer
    out = dropout(out)

    # finally send it to the final softmax dense layer
    out = dense1(out)

    # create new model with known input and output
    model = keras.models.Model(inp, out)
    
    # compile and then train
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    print('*** NEW MODEL ***')
    model.summary()

    trainable_count =  keras.utils.layer_utils.count_params(model.trainable_weights)
    
    print('adapting for {} epochs with {} trainable weights...'.format(
        ADAPT_EPOCHS, trainable_count))
    
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=ADAPT_EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test))

    for layer in orig_model.layers:
        layer.trainable = True

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
  
    trainable_count =  keras.utils.layer_utils.count_params(model.trainable_weights)

    print('fine-tuning entire network for {} epochs with {} trainable weights...'.format(
        TUNE_EPOCHS, trainable_count))
    
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=TUNE_EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test))
    
    loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
    print('Train loss={:7.5f}, accuracy={:7.5f}'.format(loss, accuracy))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test  loss={:7.5f}, accuracy={:7.5f}'.format(loss, accuracy))

    filename = 'models/cnn_xfer.h5'

    model.save(filename)
    print('wrote', filename) 

if __name__ == '__main__':
    main()

    
