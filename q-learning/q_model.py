import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, Adam


def q_model():
    model = Sequential()
    model.add(Dense(1))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('sigmoid'))
    model.add(Dense(3, activation='linear'))

    adam = Adam()
    model.compile(loss='mse', optimizer=adam, metrics = ['mse'])

    return model
