# import tensorflow as tf
from keras.layers import Dense
from tensorflow import keras


def Linear(out_dim=10, activation=None) -> keras.Model:
    return keras.Sequential((Dense(units=out_dim, activation=activation),))
