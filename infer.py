import os

import tensorflow as tf

# import tqdm # fails smh
from tqdm.autonotebook import tqdm

import names
import train_.parameters as parameters
from models import Autoencoder, Linear

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    Xy_train, (X_val, y_val) = mnist.load_data()
    X_val = X_val / 255.0
    X_val = X_val[..., tf.newaxis].astype("float32")

    batch_size = parameters.batch_size
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    if not os.path.exists(names.artifacts):
        raise FileNotFoundError(
            f"directory `{names.artifacts}` must exist and contain"
            + f"`{os.path.basename(names.ae_weights)}` after running train/autoencoder"
        )
    if not os.path.exists(names.ae_weights):
        raise FileNotFoundError(f"file `{names.ae_weights}` not found")
    if not os.path.exists(names.clsf_fc_weights):
        raise FileNotFoundError(f"file `{names.clsf_fc_weights} not found`")

    ae = Autoencoder()
    ae.build(input_shape=(batch_size, 28, 28, 1))
    ae.load_weights(names.ae_weights)
    clsf = Linear(activation=None)
    clsf.build(input_shape=(batch_size, ae.hid_dim))
    clsf.load_weights(names.clsf_fc_weights)

    acc = tf.keras.metrics.Accuracy()

    inference = []

    # can't append to list in tf.function
    # @tf.function
    def _step(images, labels):
        hidden = ae.encode(images)
        pred = clsf(hidden, training=True)
        pred = tf.math.argmax(pred, axis=1)

        inference.extend(pred)
        acc(labels, pred)

    acc.reset_states()
    for images, labels in tqdm(val_data):
        _step(images, labels)

    print(f"inference (validation) accuracy is {acc.result():.3f}")

    print(f"saving inference results into {names.clsf_inference}")
    with open(names.clsf_inference, "w") as file:
        for inf in inference:
            file.write(f"{inf}\n")
