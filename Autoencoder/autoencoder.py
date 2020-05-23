from keras.layers import Input, BatchNormalization, Dense, Conv2D, Flatten
from keras.layers import Conv2DTranspose, Reshape, Activation, LeakyReLU, ReLU, Dropout
from keras.models import Model
from keras import backend as K
import numpy as np


class Autoencoder:
    @staticmethod
    def build(width, height, depth, latent_dim=128, nfilter=32):
        input_shape = (height, width, depth)
        # 32 * 32
        input_encoder = Input(shape=input_shape)
        e = input_encoder
        # 16 * 16
        for_loop = 4
        for i in range(for_loop):
            e = Conv2D((2 ** i) * nfilter, (5, 5),
                       strides=(2, 2), padding="same")(e)
            e = LeakyReLU(0.2)(e)
            e = BatchNormalization()(e)
            e = Dropout(0.5)(e)
        e_size = K.int_shape(e)
        e = Flatten()(e)
        output_encoder = Dense(latent_dim)(e)
        encoder = Model(inputs=input_encoder,
                        outputs=output_encoder, name="encoder")

        input_decoder = Input(shape=(latent_dim,))
        d = Dense(np.prod(e_size[1:]))(input_decoder)
        d = Reshape((e_size[1], e_size[2], e_size[3]))(d)
        for i in range(for_loop):
            d = Conv2DTranspose((2 ** (for_loop - i - 1)) *
                                nfilter, (5, 5), strides=(2, 2), padding="same")(d)
            d = LeakyReLU(0.2)(d)
            d = BatchNormalization()(d)
            e = Dropout(0.5)(e)
        d = Conv2DTranspose(depth, (5, 5), strides=(1, 1), padding="same")(d)
        output_decoder = Activation("sigmoid")(d)
        decoder = Model(inputs=input_decoder,
                        outputs=output_decoder, name="decoder")
        autoencoder = Model(inputs=input_encoder, outputs=decoder(
            encoder(input_encoder)), name="autoencoder")
        return autoencoder

