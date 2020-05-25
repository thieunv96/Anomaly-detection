from keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Input, Reshape
from keras.layers import Conv2DTranspose, Activation, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.initializers import RandomNormal
import math

class Generator:
    @staticmethod
    def build(width, height, depth, latent_dim=128, nfilter=32):
        init = RandomNormal(stddev=0.02)
        
        input_generator = Input(shape=(latent_dim,))
        G = input_generator
        G = Dense(2*2*8*nfilter, use_bias=True, bias_initializer=init)(G)

        # 2*2
        G = Reshape((2, 2, 8*nfilter))(G)
        G = BatchNormalization()(G)
        G = LeakyReLU(0.2)(G)
        iteration = int(math.log2(width) - 1)
        for i in range(iteration):
            G = Conv2DTranspose(2**(iteration - i -1) * nfilter, kernel_size=(5, 5),
                                strides=(2, 2), padding="same", kernel_initializer=init)(G)
            G = BatchNormalization()(G)
            G = LeakyReLU(0.2)(G)
        G = Conv2DTranspose(depth, kernel_size=(5, 5),
                            strides=(1, 1), padding="same", kernel_initializer=init)(G)
        G = BatchNormalization()(G)
        output_generator = Activation('tanh')(G)
        model = Model(input_generator, output_generator, name="generator")
        return model

if __name__ == "__main__":
    m = Generator.build(32,32,3, nfilter=32)
    m.summary()