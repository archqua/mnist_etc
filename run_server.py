import json
import os
import subprocess
from dataclasses import dataclass, field

import hydra
import mlflow
import numpy as np
import onnx

import names
from train import ClassifierReducedConfig
from train_ import AutoencoderTrainConfig


# part of what can be found in train.py
@dataclass
class TrainConfig:
    autoencoder: AutoencoderTrainConfig = field(default_factory=AutoencoderTrainConfig)
    classifier: ClassifierReducedConfig = field(default_factory=ClassifierReducedConfig)


@dataclass
class ServerConfig:
    tracking_uri: str = "http://localhost:5000"
    train_cfg: TrainConfig = field(default_factory=TrainConfig)


@hydra.main(version_base=None, config_path="./conf", config_name="server")
def main(cfg: ServerConfig):
    onnx_model = onnx.load_model(
        names.full_model_weights(
            hid_dim=cfg.train_cfg.autoencoder.hid_dim,
            ae_epochs=cfg.train_cfg.autoencoder.epochs,
            ae_private=cfg.train_cfg.autoencoder.private,
            clsf_epochs=cfg.train_cfg.classifier.epochs,
            clsf_private=cfg.train_cfg.classifier.private,
            suffix=".onnx",
        )
    )

    # https://github.com/mlflow/mlflow/issues/7819
    mlflow.set_tracking_uri(cfg.tracking_uri)
    batch_size = 32
    inp_example = np.empty((batch_size, 28, 28, 1), dtype=np.float32)
    outp_example = np.empty((batch_size, 10), dtype=np.float32)
    signature = mlflow.models.signature.infer_signature(inp_example, outp_example)
    with mlflow.start_run():
        model_info = mlflow.onnx.log_model(onnx_model, "onnx_model", signature=signature)

    model_uri = os.path.join("mlartifacts", "0", str(model_info.run_id))
    model_info_json = {
        # this raises warning
        # "name": "mnist_ae_clsf",
        "name": "mnist_etc",
        "implementation": "mlserver_mlflow.MLflowRuntime",
        "parameters": {
            # "uri": model_info.model_uri,
            "uri": os.path.join(model_uri, "artifacts", "onnx_model"),
        },
    }
    with open("model-settings.json", "w") as fp:
        json.dump(model_info_json, fp, indent=2)
        fp.write("\n")

    print("starting server")
    subprocess.run(["mlserver", "start", "."])


if __name__ == "__main__":
    main()
