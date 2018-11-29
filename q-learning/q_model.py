import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, Adam


def q_model():
    model = Sequential()
    # Input layer
    model.add(Dense(1))

    # layers, activation per layer, dropout at a given layer, node at a given layer
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dropout(0.15))
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('relu'))

    # Output layer
    model.add(Dense(3, activation='linear'))

    adam = Adam()
    model.compile(loss='mse', optimizer=adam, metrics=['mse'])

    return model
