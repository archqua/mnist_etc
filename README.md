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
[https://www.sciencedirect.com/science/article/abs/pii/S0031320309003380](Bayessian hill climbing)
is examined in `noprotect_reconstruction.py`.
Run as `[poetry run] python3 noprotect_reconstruction.py`.


# TODO
Minor:
- fix `poetry build`

Major:
- automatic docgen with sphinx and github actions
- `tensorflow.privacy` to break hill climbing reconstruction
