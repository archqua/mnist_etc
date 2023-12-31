# import argparse
import os
import pickle
from dataclasses import dataclass

import hydra
import mlflow
import tensorflow as tf
import tensorflow_privacy as tf_privacy
import tf2onnx

# import tqdm # fails smh
from tqdm.autonotebook import tqdm

import names
import train_.parameters as parameters
from models import Autoencoder


@dataclass
class AutoencoderTrainConfig:
    hid_dim: int = 32
    private: bool = False
    epochs: int = 3


# @hydra.main(config_path = "../conf/hyperparam", config_name = "autoencoder")
def main(cfg: AutoencoderTrainConfig):
    with mlflow.start_run(nested=True, run_name="autoencoder"):
        # print(f"HYDRA CWD: {cfg.data}")
        hid_dim = cfg.hid_dim
        use_tf_privacy = cfg.private
        epochs = cfg.epochs
        mlflow.log_params(
            {
                "autoencoder latent dimension": hid_dim,
                "autoencoder private learning": use_tf_privacy,
                "autoencoder train epochs": epochs,
            }
        )

        if not os.path.exists("data"):
            raise FileNotFoundError("directory data needs to be loaded via dvc")
        X_train, y_train = pickle.load(open("data/train.pkl", "rb"))
        X_val, y_val = pickle.load(open("data/val.pkl", "rb"))

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

        ae = Autoencoder(hid_dim=hid_dim)
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

        prefix = ""  # essentially obscure
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

            # print(f"train loss: {train_loss.result()}, validation loss: {val_loss.result()}")
            mlflow.log_metrics(
                {
                    "autoencoder train loss": float(train_loss.result()),
                    "autoencoder validation loss": float(val_loss.result()),
                },
                step=epoch,
            )

            indices = tf.random.uniform(
                shape=(10,), minval=0, maxval=n_examples, dtype=tf.int32
            )
            picdir = names.ae_training_examples(
                epoch,
                hid_dim=hid_dim,
                epochs=epochs,
                private=use_tf_privacy,
                prefix=prefix,
            )
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

        ae_weights_name = names.ae_weights(
            hid_dim=hid_dim,
            epochs=epochs,
            private=use_tf_privacy,
            prefix=prefix,
        )
        print("saving autoencoder weights into " + ae_weights_name)
        ae.save_weights(ae_weights_name)
        ae_weights_name_onnx = names.ae_weights(
            hid_dim=hid_dim,
            epochs=epochs,
            private=use_tf_privacy,
            prefix=prefix,
            suffix=".onnx",
        )
        print("saving autoencoder weights into " + ae_weights_name_onnx)
        _ = tf2onnx.convert.from_keras(
            ae,
            input_signature=(tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),),
            output_path=ae_weights_name_onnx,
        )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="train autoencoder")
    # parser.add_argument(
    #     "-p",
    #     "--private",
    #     action="store_true",
    #     dest="use_tf_privacy",
    #     default=default_privacy,
    # )
    # parser.add_argument(
    #     "-e",
    #     "--epochs",
    #     dest="epochs",
    #     default=default_epochs,
    #     type=int,
    # )
    # args = parser.parse_args()
    # main(args.epochs, use_tf_privacy=args.use_tf_privacy)
    with hydra.initialize(version_base=None, config_path="../conf/hyperparam"):
        cfg = hydra.compose(config_name="autoencoder")
        main(cfg)
