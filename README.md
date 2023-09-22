# mnist_etc
This is for MLOps course.

Currently a simple autoencoder is trained on MNIST for 2 epochs
in `main.py`.
Resulting weights after two epochs are saved in `artifacts/weights.h5`
and digit images -- original and reconstructed are saved
under names `orig_{digit}.png` and `rec_{digit}.png` in `artifacts/{epoch}`.
