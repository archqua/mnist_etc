# import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Dense,
)


def Linear(out_dim=10, activation=None) -> keras.Model:
    return keras.Sequential((Dense(units=out_dim, activation=activation),))
