from dataclasses import dataclass

import hydra
import mlflow
from dvc.api import DVCFileSystem

import train_


@dataclass
class TrainConfig:
    autoencoder = train_.AutoencoderTrainConfig()
    classifier = train_.ClassifierTrainConfig()
    run_name: str = None
    tracking_uri: str = ""
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
        train_.classifier.main(cfg.classifier)

        print("training finished, results can be found in `artifacts` directory")


if __name__ == "__main__":
    main()
