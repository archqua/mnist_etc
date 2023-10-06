import os

import tensorflow as tf

# import tqdm # fails smh
from tqdm.autonotebook import tqdm

# import .names  # invalid
# import train.names as names
# import train.parameters as parameters
import train_.names as names
import train_.parameters as parameters
from models import Autoencoder, Linear


def main():
    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    X_train, X_val = X_train / 255.0, X_val / 255.0
    X_train = X_train[..., tf.newaxis].astype("float32")
    X_val = X_val[..., tf.newaxis].astype("float32")

    batch_size = parameters.batch_size
    train_data = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(10000)
        .batch(batch_size)
    )
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    ae = Autoencoder()
    ae.build(input_shape=(batch_size, 28, 28, 1))
    ae.load_weights(names.ae_weights)
    clsf = Linear(activation=None)
    lo = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean("train_loss")
    train_acc = tf.keras.metrics.Accuracy()

    @tf.function
    def _train_step(images, labels):
        hidden = ae.encode(images)
        with tf.GradientTape() as tape:
            pred = clsf(hidden, training=True)
            loss = lo(labels, pred)
        gradients = tape.gradient(loss, clsf.trainable_variables)
        opt.apply_gradients(zip(gradients, clsf.trainable_variables))

        train_loss(loss)
        train_acc(labels, tf.math.argmax(pred, axis=1))

    val_loss = tf.keras.metrics.Mean("val_loss")
    val_acc = tf.keras.metrics.Accuracy()

    @tf.function
    def _val_step(images, labels):
        hidden = ae.encode(images)
        pred = clsf(hidden, training=True)
        loss = lo(labels, pred)

        val_loss(loss)
        val_acc(labels, tf.math.argmax(pred, axis=1))

    if not os.path.exists(names.artifacts):
        raise FileNotFoundError(
            f"directory `{names.artifacts}` must exist and contain"
            + f"`{os.path.basename(names.ae_weights)}` after running train/autoencoder"
        )
    if not os.path.exists(names.ae_weights):
        raise FileNotFoundError("file `{names.ae_weights}` not found")
    EPOCHS = 2

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()

        print(f"training epoch {epoch + 1} in {EPOCHS}")
        for images, labels in tqdm(train_data):
            _train_step(images, labels)
        for images, labels in val_data:
            _val_step(images, labels)

        print(f"train loss: {train_loss.result()}, validation loss: {val_loss.result()}")
        print(
            f"train accuracy: {train_acc.result():.3f}, validation accuracy: {val_acc.result():.3f}"
        )

    print("saving classifier FC weights into " + names.clsf_fc_weights)
    clsf.save_weights(names.clsf_fc_weights)


if __name__ == "__main__":
    main()
