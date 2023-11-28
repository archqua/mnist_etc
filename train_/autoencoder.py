import argparse
import os
import pickle

import tensorflow as tf
import tensorflow_privacy as tf_privacy
from dvc.api import DVCFileSystem

# import tqdm # fails smh
from tqdm.autonotebook import tqdm

import names
import train_.parameters as parameters
from models import Autoencoder

default_epochs = 3
default_privacy = False


def main(epochs=default_epochs, use_tf_privacy=default_privacy):
    # mnist = tf.keras.datasets.mnist
    fs = DVCFileSystem()
    fs.get("data", "data", recursive=True)

    X_train, y_train = pickle.load(open("data/train.pkl", "rb"))
    X_val, y_val = pickle.load(open("data/val.pkl", "rb"))
    # (X_train, y_train), (X_val, y_val) = mnist.load_data()

    @tf.function
    def _preproc(images, labels):
        return tf.cast(images, tf.float32)[..., tf.newaxis] / 255.0, labels

    batch_size = parameters.batch_size
    train_data = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(X_train.shape[0])
        .map(_preproc, num_parallel_calls=batch_size)
        .batch(batch_size)
    )
    val_data = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .map(_preproc, num_parallel_calls=batch_size)
        .batch(batch_size)
    )

    n_examples = 7
    # val_masks = tf.constant([y_val == i for i in range(10)])
    val_examples = []
    for i in range(10):
        val_examples_i = []
        for X_batch, y_batch in val_data:
            mask = y_batch == i
            val_examples_i.extend(list(X_batch[mask]))
            if len(val_examples_i) >= n_examples:
                val_examples_i = tf.stack(val_examples_i[:n_examples])
                break
        val_examples.append(val_examples_i)
    val_examples = tf.stack(val_examples)
    # val_examples = tf.constant(
    #     [X_val[val_masks[i, ...]][:n_examples, ...] for i in range(10)]
    # )

    ae = Autoencoder()
    lo = tf.keras.losses.MeanSquaredError()
    if use_tf_privacy:
        opt = tf_privacy.DPKerasAdamOptimizer(
            l2_norm_clip=1.0,
            noise_multiplier=0.01,
            num_microbatches=1,
        )
    else:
        opt = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean("train_loss")

    @tf.function
    def _train_step(images):
        with tf.GradientTape() as tape:
            reconstruction = ae(images, training=True)
            loss = lo(images, reconstruction)
        if use_tf_privacy:
            opt.minimize(loss, ae.trainable_variables, tape=tape)
        else:
            gradients = tape.gradient(loss, ae.trainable_variables)
            opt.apply_gradients(zip(gradients, ae.trainable_variables))

        train_loss(loss)

    val_loss = tf.keras.metrics.Mean("val_loss")

    @tf.function
    def _val_step(images):
        reconstruction = ae(images, training=False)
        loss = lo(images, reconstruction)

        val_loss(loss)

    prefix = "tf_private_" if use_tf_privacy else ""
    if not os.path.exists(names.artifacts):
        os.mkdir(names.artifacts)

    for epoch in range(epochs):
        print(f"training epoch {epoch + 1} in {epochs}")
        train_loss.reset_states()
        val_loss.reset_states()

        for images, labels in tqdm(train_data):
            _train_step(images)
        for images, labels in val_data:
            _val_step(images)

        print(f"train loss: {train_loss.result()}, validation loss: {val_loss.result()}")

        indices = tf.random.uniform(
            shape=(10,), minval=0, maxval=n_examples, dtype=tf.int32
        )
        picdir = names.ae_training_examples(epoch, prefix=prefix)
        if not os.path.exists(picdir):
            # TODO handle permission denied?
            os.makedirs(picdir)
        print("saving examples into " + picdir)
        for digit, index in zip(range(10), indices):
            img = val_examples[digit : digit + 1, index, ...]
            rec = ae(img)
            tf.keras.utils.save_img(
                os.path.join(picdir, f"orig_{digit}.png"),
                img[0, ...],
            )
            tf.keras.utils.save_img(
                os.path.join(picdir, f"rec_{digit}.png"),
                rec[0, ...],
            )

    print("saving autoencoder weights into " + names.ae_weights(prefix=prefix))
    ae.save_weights(names.ae_weights(prefix=prefix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train autoencoder")
    parser.add_argument(
        "-p",
        "--private",
        action="store_true",
        dest="use_tf_privacy",
        default=default_privacy,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        default=default_epochs,
        type=int,
    )
    args = parser.parse_args()
    main(args.epochs, use_tf_privacy=args.use_tf_privacy)
