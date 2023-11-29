from dataclasses import dataclass

import hydra
from dvc.api import DVCFileSystem

import train_


@dataclass
class TrainConfig:
    autoencoder = train_.AutoencoderTrainConfig()
    classifier = train_.ClassifierTrainConfig()
    # TODO verbosity, dvc call


@hydra.main(version_base=None, config_path="conf/hyperparam", config_name="train")
def main(cfg: TrainConfig):
    fs = DVCFileSystem()
    fs.get("data", "data", recursive=True)

    print("training autoencoder")
    train_.autoencoder.main(cfg.autoencoder)

    print("training classifier")
    train_.classifier.main(cfg.classifier)

    print("training finished, results can be found in `artifacts` directory")


if __name__ == "__main__":
    main()
