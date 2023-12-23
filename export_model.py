import os
from dataclasses import dataclass, field

import hydra
import tensorflow as tf
import tf2onnx

import names
from models import FullModel
from train import ClassifierReducedConfig
from train_ import AutoencoderTrainConfig


@dataclass
class ExportConfig:
    model_path: str = "model_repository/onnx-mnist_etc"
    version: str = "latest"
    autoencoder: AutoencoderTrainConfig = field(default_factory=AutoencoderTrainConfig)
    classifier: ClassifierReducedConfig = field(default_factory=ClassifierReducedConfig)


@hydra.main(version_base=None, config_path="conf", config_name="model_export")
def main(cfg: ExportConfig):
    prefix = ""  # essentially obscure
    full_model_weights_name = names.full_model_weights(
        hid_dim=cfg.autoencoder.hid_dim,
        ae_epochs=cfg.autoencoder.epochs,
        ae_private=cfg.autoencoder.private,
        clsf_epochs=cfg.classifier.epochs,
        clsf_private=cfg.classifier.private,
        prefix=prefix,
    )
    if not os.path.exists(names.artifacts):
        raise FileNotFoundError(
            f"directory `{names.artifacts}` must exist and contain"
            + f"`{os.path.basename(full_model_weights_name)}` after running train/autoencoder"
        )
    if not os.path.exists(full_model_weights_name):
        raise FileNotFoundError(f"file `{full_model_weights_name}` not found")

    batch_size = None
    full_model = FullModel(
        hid_dim=cfg.autoencoder.hid_dim,
        # this is problematic
        # dec_out_activation=ae.dec_out_activation,
        n_classes=10,
    )
    full_model.build(input_shape=(batch_size, 28, 28, 1))
    full_model.load_weights(full_model_weights_name)

    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"directory `{cfg.model_path}` not found")
    if not os.path.exists(os.path.join(cfg.model_path, cfg.version)):
        os.mkdir(os.path.join(cfg.model_path, cfg.version))

    _ = tf2onnx.convert.from_keras(
        full_model,
        input_signature=(tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),),
        output_path=os.path.join(cfg.model_path, cfg.version, "model.onnx"),
    )


if __name__ == "__main__":
    main()
