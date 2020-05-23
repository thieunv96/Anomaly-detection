from Autoencoder.autoencoder import Autoencoder
import utils
from keras.layers import Input, BatchNormalization, Dense, Conv2D, Flatten
from keras.layers import Conv2DTranspose, Reshape, Activation, LeakyReLU, ReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from matplotlib import pyplot as plt


width, height, depth = 32 , 32 , 3
latent_dim = 128
nfilters = 32


def train_autoencoder():
    x_train, y_train = utils.get_data_train("./data/antenna_a12/positive", width, height, depth, limit=10000)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
    opt = Adam(learning_rate=1e-4, decay=1e-4/500)
    model = Autoencoder.build(width, height, depth, latent_dim, nfilters)
    model.compile(optimizer=opt, loss="mse")
    H =model.fit(x_train, y_train, 64, 500, validation_data=(x_test, y_test))
    model.save_weights("./trained/autoencoder.hdf5")
    print(H)

def load_autoencoder():
    model = Autoencoder.build(width, height, depth, latent_dim, nfilters)
    model.load_weights("./trained/autoencoder.hdf5")
    return model

def test_autoencoder(url):
    model = load_autoencoder()
    x, _ = utils.get_data_train(url, width, height, depth, limit=1, augmentation=False)
    result = model.predict(x)[0]
    result = (result * 255.0).astype("uint8")
    x = (x[0] * 255.0).astype("uint8")
    result = np.hstack([x, result])
    # cv2.imwrite("test.png", result)
    # cv2.waitKey(0)


def get_mse():
    x, _ = utils.get_data_train(r"./data/antenna_a12/positive", width, height, depth, limit=100000, augmentation=False)
    model = load_autoencoder()
    result = model.predict(x)
    errors = []
    for images, recon in zip(x, result):
        mse = np.mean((images - recon) ** 2)
        if(mse < 0.05):
            errors.append(mse)
    
    print(min(errors), max(errors))
    plt.plot(errors)
    plt.show()
    return min(errors), max(errors)


if __name__ == "__main__":
    # train_autoencoder()
    # test_autoencoder("./data/antenna_a12/negative/321.jpg")
    get_mse()