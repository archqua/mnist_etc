# mnist_etc
This is for MLOps course.

Currently there are two train loops in `train_` directory:
`autoencoder.py` and `classifier.py` to train
a simple convolution-deconvolution autoencoder and
a semi-linear classifier, based on autoencoder's hidden representation.
These should be run as `[poetry run] python3 -m train_.{loop}`
from project's root directory or via `[poetry run] python3 train.py`.
Run classifier's training only when autoencoder's training has finished.

Resulting weights after two epochs are saved in `artifacts/ae_weights.h5`
and `artifacts/clsf_fc_weights.h5`.
During autoencoder's training example digit images -- original and reconstructed --
are saved under names `orig_{digit}.png` and `rec_{digit}.png` in
`artifacts/ae_training_examples_{epoch}`.

Inference is done via running `[poetry run] python3 infer.py` and
results are saved into `artifacts/clsf_inference.csv`.

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


# TODO
Minor:
- fix `poetry build`

Major:
- automatic docgen with sphinx and github actions
- use `hydra` for parameterization
- `tensorflow.privacy` to break hill climbing reconstruction
