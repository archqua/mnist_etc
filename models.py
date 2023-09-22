import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Reshape, Resizing
UpConv2D = Conv2DTranspose


class Encoder(keras.Model):
    def __init__(self):
        super().__init__()
        # 28 -> 24
        self.conv1 = Conv2D(8, 5, activation='relu')
        # 24 -> 12
        self.pool1 = MaxPooling2D((2, 2))
        # 12 -> 10
        self.conv2 = Conv2D(16, 3, activation='relu')
        # 10 -> 5
        self.pool2 = MaxPooling2D((2, 2))
        # 5 -> 3
        self.conv3 = Conv2D(32, 3, activation='relu')
        # 3 -> 1
        self.conv4 = Conv2D(64, 3, activation='relu')
        self.flatten = Flatten()

    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        return self.flatten(x)


class Decoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.unflatten = Reshape((1, 1, -1))
        self.uconv1 = UpConv2D(32, 3, activation='relu')
        self.uconv2 = UpConv2D(16, 3, activation='relu')
        self.upool3 = UpConv2D(16, 2, strides=2, activation='relu')
        self.uconv3 = UpConv2D(8, 3, activation='relu')
        self.upool4 = UpConv2D(8, 2, strides=2, activation='relu')
        self.uconv4 = UpConv2D(1, 5, activation='relu')

    def call(self, x):
        x = self.uconv1(self.unflatten(x))
        x = self.uconv2(x)
        x = self.uconv3(self.upool3(x))
        return self.uconv4(self.upool4(x))


class Autoencoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x):
        return self.decoder(self.encoder(x))
