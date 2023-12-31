# import argparse
import os
import pickle
from dataclasses import dataclass, field

import hydra
import mlflow
import tensorflow as tf
import tensorflow_privacy as tf_privacy
import tf2onnx

# import tqdm # fails smh
from tqdm.autonotebook import tqdm

import names
import train_.parameters as parameters
from models import Autoencoder, FullModel, Linear
from train_.autoencoder import AutoencoderTrainConfig


@dataclass
class ClassifierTrainConfig:
    inp_dim: int = 32
    private: bool = False
    epochs: int = 6
    ae_cfg: AutoencoderTrainConfig = field(default_factory=AutoencoderTrainConfig)


# @hydra.main(config_path = "../conf/hyperparam", config_name = "classifier")
def main(cfg: ClassifierTrainConfig):
    assert (
        cfg.inp_dim == cfg.ae_cfg.hid_dim
    ), "classifier's input dim must be equal to AE's hidden dim"
    with mlflow.start_run(nested=True, run_name="classifier"):
        inp_dim = cfg.inp_dim
        use_tf_privacy = cfg.private
        epochs = cfg.epochs
        mlflow.log_params(
            {
                "classifier input dimension": inp_dim,
                "classifier private learning": use_tf_privacy,
                "classifier train epochs": epochs,
            }
        )

        if not os.path.exists("data"):
            raise FileNotFoundError("directory data needs to be loaded via dvc")
        X_val, y_val = pickle.load(open("data/val.pkl", "rb"))

        @tf.function
        def _preproc(images, labels):
            return tf.cast(images, tf.float32)[..., tf.newaxis] / 255.0, labels

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
        # train_data = (
        #     tf.data.Dataset.from_tensor_slices((X_train, y_train))
        #     .shuffle(X_train.shape[0])
        #     .map(_preproc, num_parallel_calls=batch_size)
        #     .batch(batch_size)
        # )
        # val_data = (
        #     tf.data.Dataset.from_tensor_slices((X_val, y_val))
        #     .map(_preproc, num_parallel_calls=batch_size)
        #     .batch(batch_size)
        # )

        prefix = ""  # essentially obscure
        ae_weights_name = names.ae_weights(
            hid_dim=cfg.ae_cfg.hid_dim,
            epochs=cfg.ae_cfg.epochs,
            private=cfg.ae_cfg.private,
            prefix=prefix,
        )
        if not os.path.exists(names.artifacts):
            raise FileNotFoundError(
                f"directory `{names.artifacts}` must exist and contain"
                + f"`{os.path.basename(ae_weights_name)}` after running train/autoencoder"
            )
        if not os.path.exists(ae_weights_name):
            raise FileNotFoundError(f"file `{ae_weights_name}` not found")

        ae = Autoencoder(hid_dim=inp_dim)
        ae.build(input_shape=(batch_size, 28, 28, 1))
        ae.load_weights(ae_weights_name)
        clsf = Linear(activation=None)
        if use_tf_privacy:
            lo = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.losses.Reduction.NONE,
            )
            _privacy_scale = batch_size
            opt = tf_privacy.DPKerasAdamOptimizer(
                l2_norm_clip=1.0,
                noise_multiplier=0.025 * _privacy_scale,
                num_microbatches=1 * _privacy_scale,
            )
        else:
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
            if use_tf_privacy:
                opt.minimize(loss, clsf.trainable_variables, tape=tape)
            else:
                gradients = tape.gradient(loss, clsf.trainable_variables)
                opt.apply_gradients(zip(gradients, clsf.trainable_variables))

            train_loss(loss)
            train_acc(labels, tf.math.argmax(pred, axis=1))

        val_loss = tf.keras.metrics.Mean("val_loss")
        val_acc = tf.keras.metrics.Accuracy()

        @tf.function
        def _val_step(images, labels):
            hidden = ae.encode(images)
            pred = clsf(hidden, training=False)
            loss = lo(labels, pred)

            val_loss(loss)
            val_acc(labels, tf.math.argmax(pred, axis=1))

        for epoch in range(epochs):
            train_loss.reset_states()
            train_acc.reset_states()
            val_loss.reset_states()
            val_acc.reset_states()

            print(f"training epoch {epoch + 1} in {epochs}")
            for images, labels in tqdm(train_data):
                _train_step(images, labels)
            for images, labels in val_data:
                _val_step(images, labels)

            mlflow.log_metrics(
                {
                    "classifier train loss": float(train_loss.result()),
                    "classifier validation loss": float(val_loss.result()),
                },
                step=epoch,
            )
            # print(f"train loss: {train_loss.result()}, validation loss: {val_loss.result()}")
            mlflow.log_metrics(
                {
                    "classifier train accuracy": float(train_acc.result()),
                    "classifier validation accuracy": float(val_acc.result()),
                },
                step=epoch,
            )
            # print(
            #     f"train accuracy: {train_acc.result():.3f}, validation accuracy: {val_acc.result():.3f}"
            # )

            # mlflow.tensorflow.log_model(clsf, "clsf_fc")

        clsf_fc_weights_name = names.clsf_fc_weights(
            inp_dim=inp_dim,
            ae_epochs=cfg.ae_cfg.epochs,
            ae_private=cfg.ae_cfg.private,
            epochs=epochs,
            private=use_tf_privacy,
            prefix=prefix,
        )
        print("saving classifier FC weights into " + clsf_fc_weights_name)
        clsf.save_weights(clsf_fc_weights_name)
        clsf_fc_weights_name_onnx = names.clsf_fc_weights(
            inp_dim=inp_dim,
            ae_epochs=cfg.ae_cfg.epochs,
            ae_private=cfg.ae_cfg.private,
            epochs=epochs,
            private=use_tf_privacy,
            prefix=prefix,
            suffix=".onnx",
        )
        print("saving classifier FC weights into " + clsf_fc_weights_name_onnx)
        _ = tf2onnx.convert.from_keras(
            clsf,
            input_signature=(tf.TensorSpec((None, inp_dim), tf.float32, name="input"),),
            output_path=clsf_fc_weights_name_onnx,
        )

        full_model = FullModel(
            hid_dim=inp_dim, dec_out_activation=ae.dec_out_activation, n_classes=10
        )
        full_model.autoencoder = ae
        full_model.fc = clsf
        full_model.build(input_shape=(batch_size, 28, 28, 1))
        full_model_weights_name = names.full_model_weights(
            hid_dim=inp_dim,
            ae_epochs=cfg.ae_cfg.epochs,
            ae_private=cfg.ae_cfg.private,
            clsf_epochs=epochs,
            clsf_private=use_tf_privacy,
            prefix=prefix,
        )
        print("saving full model weights into " + full_model_weights_name)
        full_model.save_weights(full_model_weights_name)
        full_model_weights_name_onnx = names.full_model_weights(
            hid_dim=inp_dim,
            ae_epochs=cfg.ae_cfg.epochs,
            ae_private=cfg.ae_cfg.private,
            clsf_epochs=epochs,
            clsf_private=use_tf_privacy,
            prefix=prefix,
            suffix=".onnx",
        )
        print("saving full model weights into " + full_model_weights_name_onnx)
        _ = tf2onnx.convert.from_keras(
            full_model,
            input_signature=(tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),),
            output_path=full_model_weights_name_onnx,
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
        cfg = hydra.compose(config_name="classifier")
        main(cfg)
