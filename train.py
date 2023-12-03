from dataclasses import dataclass, field

import hydra
import mlflow
from dvc.api import DVCFileSystem

import train_


@dataclass
class ClassifierReducedConfig:
    private: bool = False
    epochs: int = 6


@dataclass
class TrainConfig:
    autoencoder: train_.AutoencoderTrainConfig = field(
        default_factory=train_.AutoencoderTrainConfig
    )
    classifier: ClassifierReducedConfig = field(default_factory=ClassifierReducedConfig)
    run_name: str = None
    # https://github.com/mlflow/mlflow/issues/7819
    tracking_uri: str = "http://localhost:5000"
    # TODO verbosity, dvc call


@hydra.main(version_base=None, config_path="conf/hyperparam", config_name="train")
def main(cfg: TrainConfig):
    # mlflow.log_param("git commit ID", mlflow.source.git.commit)
    # print(mlflow.source.git.commit)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    with mlflow.start_run(run_name=cfg.run_name):
        fs = DVCFileSystem()
        fs.get("data", "data", recursive=True)

        print("training autoencoder")
        train_.autoencoder.main(cfg.autoencoder)

        print("training classifier")
        clsf_cfg = train_.ClassifierTrainConfig(
            inp_dim=cfg.autoencoder.hid_dim,
            private=cfg.classifier.private,
            epochs=cfg.classifier.epochs,
            ae_cfg=cfg.autoencoder,
        )
        train_.classifier.main(clsf_cfg)

        print("training finished, results can be found in `artifacts` directory")


if __name__ == "__main__":
    main()
