from fAnoGAN.Generator import Generator
from fAnoGAN.Discriminator import Discriminator
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
from keras.utils import Progbar


width, height, depth = 32 , 32 , 3
latent_dim = 128
nfilters = 32


def train_gan():
    epochs = 2000
    batch_size = 64
    g = Generator.build(width, height, depth, latent_dim=latent_dim, nfilter=nfilters)
    d = Discriminator.build(width, height, depth, nfilter=nfilters)
    d.compile(optimizer=Adam(learning_rate=2e-4, beta_1=0.5), loss="binary_crossentropy")
    d.trainable = False
    z = Input(shape=(latent_dim,))
    g2d = Model(inputs=z, outputs=d(g(z)))
    g2d.compile(optimizer=Adam(learning_rate=2e-4, beta_1=0.5), loss="binary_crossentropy")

    x_train = utils.get_data_train_tanh("./data/antenna_a12/positive", width, height, depth, limit=3000, augmentation=False)[0]
    print(x_train.shape)
    real_labels = np.ones(batch_size)
    fake_labels = np.zeros(batch_size)
    z_fixed = np.random.uniform(-1, 1, size=(16, latent_dim))
    eps = 0
    loss_d= []
    loss_g = []
    # Step 2: train GAN
    for e in range(0, epochs):
        eps += 1
        steps = x_train.shape[0]//batch_size
        progress = Progbar(target=steps)
        for st in range(steps):
            d.trainable = True
            real_images = x_train[st*batch_size:(st+1)*batch_size]
            loss_d_real = d.train_on_batch(x=real_images, y=real_labels)
            z = np.random.uniform(-1, 1, size=(batch_size, latent_dim))
            fake_images = g.predict_on_batch(z)
            loss_d_fake = d.train_on_batch(x=fake_images, y=fake_labels)
            loss_d.append(loss_d_real + loss_d_fake)
            d.trainable = False
            lg = g2d.train_on_batch(x=z, y=real_labels)
            loss_g.append(lg) 
            progress.update(st, values=[
                            ("Epochs", eps), ("loss_g", loss_g[-1]), ("loss_d", loss_d[-1])])
            if(st == 0 and (e + 1) % 20 == 0):
                img_generated = g.predict(z_fixed)
                img_generated = (img_generated + 1) * 127.5
                result = np.hstack(img_generated)
                cv2.imwrite("./results/e{}.png".format(e+1), result)
    g.save_weights("./trained/g.hdf5")
    g2d.save_weights("./trained/g2d.hdf5")
    print("[INFO] Train GAN finish...")
    plt.plot(loss_g)
    plt.plot(loss_d)
    plt.savefig("loss_GAN.png")
    

if __name__ == "__main__":
    train_gan()

