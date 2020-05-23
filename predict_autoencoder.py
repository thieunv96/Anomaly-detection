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
import glob
from matplotlib import pyplot as plt


width, height, depth = 32 , 32 , 3
latent_dim = 128
nfilters = 32

def predict(model, img, min_thresh, max_thresh):
    x = np.asarray([img])
    x = x.astype(np.float32)/255.
    result = model.predict(x)[0]
    mse = np.mean((result - x[0]) ** 2)
    return mse

def load_autoencoder():
    model = Autoencoder.build(width, height, depth, latent_dim, nfilters)
    model.load_weights("./trained/autoencoder.hdf5")
    return model


def validate(url):
    min_thresh = 0.0005164906 #0.00037664003 #0.0010326569
    max_thresh = 0.013614137 #0.0059956726 #0.0028067187
    files = glob.glob(url)
    model = load_autoencoder()
    errors = 0
    loss = []
    for f in files:
        img = cv2.imread(f, 1)
        img = cv2.resize(img, (height, width))
        mse = predict(model, img, min_thresh, max_thresh)
        loss.append(mse)
        if(mse > max_thresh or mse < min_thresh):
            pass
        else:
            errors += 1
    print("[INFO] Fail {} / {} images,  Accuracy = {}".format(errors, len(files), (1- errors/ len(files))*100))
    loss  = np.asarray(loss)
    return loss


def show_dialog():
    loss = validate(r"D:\Heal\Projects\NBB VI A12\Dataset\antenna\3\*.*")
    thresh = np.zeros(len(loss), dtype=np.float32)
    thresh += 0.010691941
    plt.plot(loss)
    loss = validate(r"./data/antenna_a12/negative/*.*")
    plt.plot(loss)
    plt.plot(thresh)
    plt.show()

if __name__ == "__main__":
    show_dialog()