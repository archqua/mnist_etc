import pickle

import tensorflow as tf

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    Xy_train, Xy_val = mnist.load_data()

    pickle.dump(Xy_train, open("data/train.pkl", "wb"))
    pickle.dump(Xy_val, open("data/val.pkl", "wb"))
