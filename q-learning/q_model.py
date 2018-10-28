import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam


def q_model():
    tsteps = 1
    batch_size = 1
    num_features = 2

    model = Sequential()
    model.add(LSTM(64,
                   input_shape=(1, num_features),
                   return_sequences=True,
                   stateful=False))
    model.add(Dropout(0.5))

    model.add(LSTM(64,
                   input_shape=(1, num_features),
                   return_sequences=False,
                   stateful=False))
    model.add(Dropout(0.5))

    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

    adam = Adam()
    model.compile(loss='mse', optimizer=adam)

    return model
