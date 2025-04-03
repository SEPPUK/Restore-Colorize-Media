from keras import backend as K
from keras.engine import Layer

class FusionLayer(Layer):
    def call(self, inputs, mask=None):
        imgTensors, embTensors = inputs
        reshapedShape = imgTensors.shape[:3].concatenate(embTensors.shape[1])
        embTensors = K.repeat(embTensors, imgTensors.shape[1] * imgTensors.shape[2])
        embTensors = K.reshape(embTensors, reshapedShape)
        return K.concatenate([imgTensors, embTensors], axis=3)

    def compute_output_shape(self, inputShapes):
        # Must have 2 tensors as input
        assert inputShapes and len(inputShapes) == 2
        imgShapes, embShapes = inputShapes

        # The batch size of the two tensors must match
        assert imgShapes[0] == embShapes[0]

        # (batchSize, width, height, embeddingLen + depth)
        return imgShapes[:3] + (imgShapes[3] + embShapes[1],)