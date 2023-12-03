from typing import Optional

from tensorflow import keras

from .autoencoder import Autoencoder, Decoder, Encoder
from .classifier import Linear


class FullModel(keras.Model):
    def __init__(
        self,
        hid_dim: int = 32,
        dec_out_activation: Optional[str] = "relu",
        n_classes: int = 10,
    ):
        super().__init__()
        self.autoencoder = Autoencoder(
            hid_dim=hid_dim, dec_out_activation=dec_out_activation
        )
        self.fc = Linear(out_dim=n_classes)

    def call(self, X):
        return self.fc(self.autoencoder.encode(X))

    def encode(self, X):
        return self.autoencoder.encode(X)

    def decode(self, X):
        return self.autoencoder.decode(X)
