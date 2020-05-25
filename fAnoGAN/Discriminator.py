from keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Input, Reshape
from keras.layers import Conv2DTranspose, Activation, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.initializers import RandomNormal
import math

class Discriminator:
    @staticmethod
    def build(width, height, depth, nfilter=32):
        init = RandomNormal(stddev=0.02)
        
        input_disc = Input(shape=(height, width, depth))
        D = input_disc
        iteration = int(math.log2(width) - 1)
        for i in range(iteration):
            D = Conv2D(2**i * nfilter, kernel_size=(5, 5),
                                strides=(2, 2), padding="same", kernel_initializer=init)(D)
            D = BatchNormalization()(D)
            D = LeakyReLU(0.2)(D)
        D = Flatten()(D)
        D = Dense(1)(D)
        output_disc= Activation('sigmoid')(D)
        model = Model(input_disc, output_disc, name="discriminator")
        return model

if __name__ == "__main__":
    m = Discriminator.build(32,32,3, nfilter=32)
    m.summary()