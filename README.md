# mnist_etc
This is for MLOps course.

Currently there are two train loops in `train_` directory:
`autoencoder.py` and `classifier.py` to train
a simple convolution-deconvolution autoencoder and
a semi-linear classifier, based on autoencoder's hidden representation.
These should be run as `[poetry run] python3 -m train.{loop}`
from project's root directory or via `[poetry run] python3 train.py`.
Run classifier's training only when autoencoder's training has finished.

Resulting weights after two epochs are saved in `artifacts/ae_weights.h5`
and `artifacts/clsf_fc_weights.h5`.
During autoencoder's training example digit images -- original and reconstructed --
are saved under names `orig_{digit}.png` and `rec_{digit}.png` in
`artifacts/ae_training_examples_{epoch}`.


# TODO
Minor:
- fix `poetry build`

Major:
- automatic docgen with sphinx and github actions
- hill climbing reconstruction based on certainty levels
- `tensorflow.privacy` to break hill climbing reconstruction
