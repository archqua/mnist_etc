import os
import pickle

import hydra
import tensorflow as tf

# import dvc.api
from dvc.api import DVCFileSystem

# import tqdm # fails smh
from tqdm.autonotebook import tqdm

import names
import train_.parameters as parameters
from models import Autoencoder, Linear
from train import TrainConfig

InferenceConfig = TrainConfig


@hydra.main(version_base=None, config_path="conf/hyperparam", config_name="infer")
def main(cfg: InferenceConfig):
    fs = DVCFileSystem()
    fs.get("data", "data", recursive=True)
    X_val, y_val = pickle.load(open("data/val.pkl", "rb"))

    @tf.function
    def _preproc(images, labels):
        return tf.cast(images, tf.float32)[..., tf.newaxis] / 255.0, labels

    batch_size = parameters.batch_size
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    batch_size = parameters.batch_size
    dataset = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .shuffle(X_val.shape[0])
        .map(_preproc, num_parallel_calls=batch_size)
    )
    left_size = int((dataset.cardinality().numpy() * 0.9) // 32 * 32)
    train_data, val_data = tf.keras.utils.split_dataset(dataset, left_size=left_size)
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    ae_weights_name = names.ae_weights(
        hid_dim=cfg.autoencoder.hid_dim,
        epochs=cfg.autoencoder.epochs,
        private=cfg.autoencoder.private,
    )
    clsf_fc_weights_name = names.clsf_fc_weights(
        inp_dim=cfg.autoencoder.hid_dim,
        ae_epochs=cfg.autoencoder.epochs,
        ae_private=cfg.autoencoder.private,
        epochs=cfg.classifier.epochs,
        private=cfg.classifier.private,
    )
    if not os.path.exists(names.artifacts):
        raise FileNotFoundError(
            f"directory `{names.artifacts}` must exist and contain"
            + f"`{os.path.basename(names.ae_weights())}` after running train/autoencoder"
        )
    if not os.path.exists(ae_weights_name):
        raise FileNotFoundError(f"file `{ae_weights_name}` not found")
    if not os.path.exists(clsf_fc_weights_name):
        raise FileNotFoundError(f"file `{clsf_fc_weights_name} not found`")

    ae = Autoencoder()
    ae.build(input_shape=(batch_size, 28, 28, 1))
    ae.load_weights(ae_weights_name)
    clsf = Linear(activation=None)
    clsf.build(input_shape=(batch_size, ae.hid_dim))
    clsf.load_weights(clsf_fc_weights_name)

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

    clsf_inference_name = names.clsf_inference(
        inp_dim=cfg.autoencoder.hid_dim,
        ae_epochs=cfg.autoencoder.epochs,
        ae_private=cfg.autoencoder.private,
        epochs=cfg.classifier.epochs,
        private=cfg.classifier.private,
    )
    print(f"saving inference results into {clsf_inference_name}")
    with open(clsf_inference_name, "w") as file:
        for inf in inference:
            file.write(f"{inf}\n")


if __name__ == "__main__":
    main()
