#!/usr/bin/env python3

import os

import tensorflow as tf
import models
# import tqdm # fails smh
from tqdm.autonotebook import tqdm

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    X_train, X_val = X_train / 255.0, X_val / 255.0
    X_train = X_train[..., tf.newaxis].astype('float32')
    X_val = X_val[..., tf.newaxis].astype('float32')

    n_examples = 7
    # train_masks = tf.constant([y_train == i for i in range(10)])
    # train_examples = tf.constant([X_train[train_masks[i, ...]][:n_examples, ...] for i in range(10)])
    val_masks = tf.constant([y_val == i for i in range(10)])
    val_examples = tf.constant([X_val[val_masks[i, ...]][:n_examples, ...] for i in range(10)])

    batch_size = 32
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(batch_size)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    ae = models.Autoencoder()
    lo = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean('train_loss')
    @tf.function
    def train_step(images):
        with tf.GradientTape() as tape:
            reconstruction = ae(images, training=True)
            loss = lo(images, reconstruction)
        gradients = tape.gradient(loss, ae.trainable_variables)
        opt.apply_gradients(zip(gradients, ae.trainable_variables))

        train_loss(loss)

    val_loss = tf.keras.metrics.Mean('val_loss')
    @tf.function
    def val_step(images):
        reconstruction = ae(images, training=True)
        loss = lo(images, reconstruction)

        val_loss(loss)


    if not os.path.exists("artifacts"):
        os.mkdir("artifacts")
    EPOCHS = 2

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        val_loss.reset_states()

        print(f"training epoch {epoch + 1} in {EPOCHS}")
        for images, labels in tqdm(train_data):
            train_step(images)
        for images, labels in val_data:
            val_step(images)

        print(f"train loss: {train_loss.result()}, validation loss: {val_loss.result()}")

        indices = tf.random.uniform(shape=(10,), minval=0, maxval=n_examples, dtype=tf.int32)
        picdir = f"artifacts/{epoch+1}"
        if not os.path.exists(picdir):
            # TODO handle permission denied?
            os.makedirs(picdir)
        print("saving examples into " + picdir)
        for digit, index in zip(range(10), indices):
            img = val_examples[digit:digit+1, index, ...]
            rec = ae(img)
            tf.keras.utils.save_img(f"artifacts/{epoch+1}/orig_{digit}.png", img[0, ...])
            tf.keras.utils.save_img(f"artifacts/{epoch+1}/rec_{digit}.png", rec[0, ...])

    print("saving autoencoder weights into artifacts/weights.h5")
    ae.save_weights("artifacts/weights.h5")


