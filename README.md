# mnist_etc
This is for MLOps course.

Configuration is stored via `dvc`.
Run `dvc pull conf.dvc` before doing anything.

Currently there are two train loops in `train_` directory:
`autoencoder.py` and `classifier.py` to train
a simple convolution-deconvolution autoencoder and
a semi-linear classifier, based on autoencoder's hidden representation.
These should be run as `[poetry run] python3 -m train_.<loop> [key=value list of hydra settings]`
from project's root directory or via `[poetry run] python3 train.py [key=value list of hydra settings]`.
Run classifier's training only when autoencoder's training has finished.

Resulting weights after two epochs are saved in `artifacts/ae_h<hid_dim>_e<epochs>[_private]_weights.<format>`
and `artifacts/clsf_h<inp_dim>_ee<AE epochs>[_eprivate]_e<epochs>[_private]_fc_weights.<format>`
with format being either `h5` or `onnx`.
During autoencoder's training example digit images -- original and reconstructed --
are saved under names `orig_{digit}.png` and `rec_{digit}.png` in
`artifacts/ae_h<hid_dim>_e<epochs>[_private]_examples_{epoch}`.

Inference is done via running `[poetry run] python3 infer.py [key=value list of hydra settings]` and
results are saved into `artifacts/clsf_h<inp_dim>_ee<AE epochs>[_eprivate]_e<epochs>[_private]_inference.csv`.

Reconstruction via
[Bayessian hill climbing](https://www.sciencedirect.com/science/article/abs/pii/S0031320309003380)
is examined in `noprotect_reconstruction.py`.
Run as `[poetry run] python3 noprotect_reconstruction.py`.


# Dependencies

## Build
- `poetry`

## Run
- `python` version `>=3.9,<3.12`
- `tensorflow` version `<2.14.0`
- `tqdm` version `^4.66.1`
- `pillow` version `^10.0.1`
- `tensorflow-privacy` version `<0.8.11`
- `dvc` version `^3.30.1`
- `dvc-gdrive` version `^2.20.0`
- `hydra-core` version `^1.3.2`
- `mlflow` version `^2.8.1`
- `onnx` version `<1.15.0`
- `tf2onnx` version `^1.15.1`


# TODO
Minor:
- fix `poetry build`

Major:
- automatic docgen with sphinx and github actions
- `tensorflow.privacy` to break hill climbing reconstruction (it doesn't break reconstruction)
- remove tesnorflow import when running inference server
