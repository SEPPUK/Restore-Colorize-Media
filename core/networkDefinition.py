from keras.engine import InputLayer
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential

from .fusionLayer import FusionLayer


class Colorization:
    def __init__(self, depthAfterFusion):
        self.encoder = _buildEncoder()
        self.fusion = FusionLayer()
        self.afterFusion = Conv2D(depthAfterFusion, (1, 1), activation="relu")
        self.decoder = _buildDecoder(depthAfterFusion)

    def build(self, imgL, imgEmb):
        imgEnc = self.encoder(imgL)

        fusion = self.fusion([imgEnc, imgEmb])
        fusion = self.afterFusion(fusion)

        return self.decoder(fusion)


def _buildEncoder():
    model = Sequential(name="encoder")
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    return model


def _buildDecoder(encodingDepth):
    model = Sequential(name="decoder")
    model.add(InputLayer(input_shape=(None, None, encodingDepth)))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(2, (3, 3), activation="tanh", padding="same"))
    model.add(UpSampling2D((2, 2)))
    return model
