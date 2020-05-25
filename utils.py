import cv2
import shutil
import glob
import numpy as np


def get_data(src, dst, limit=1000):
    files = glob.glob("{}/*.png".format(src))
    files += glob.glob("{}/*.jpg".format(src))
    files = files[:limit]
    for i, f in enumerate(files):
        if ((i+1) % 1000 == 0):
            print("[INFO] Download {} items finished!".format(i+1))
        shutil.copy(f, "{}/{}.jpg".format(dst, i))
    print("[INFO] Download {} items finished!".format(len(files)))


def get_data_train(url, width, height, depth, limit=1000, augmentation = True):
    
    files = glob.glob("{}/*.png".format(url))
    if("jpg" in url or "png" in url):
        files += glob.glob("{}".format(url))
    files += glob.glob("{}/*.jpg".format(url))
    files = files[:limit]
    x = []
    y = []
    for i, f in enumerate(files):
        if ((i+1) % 1000 == 0):
            print("[INFO] Download {} items finished!".format(i+1))
        img = cv2.imread(f, depth // 3)
        img = cv2.resize(img, (height, width))
        x.append(img)
        y.append(img.copy())
    if(augmentation):
        for i in range(len(x)):
            img = x[i]
            shape = img.shape
            for _ in range(10):
                ct = (int(np.random.rand(1) * shape[1]), int(np.random.rand(1) * shape[0]))
                img = cv2.rectangle(img, (ct[0]-2, ct[1]-2), (ct[0]+2, ct[1] + 2), (0,0,0), -1)
            x.append(img)
            y.append(y[i].copy())

    x = np.asarray(x)
    x = x.astype("float32") / 255.
    x = np.expand_dims(x, axis=-1) if(depth // 3 == 0) else x
    y = np.asarray(y)
    y = y.astype("float32") / 255.
    y = np.expand_dims(y, axis=-1) if(depth // 3 == 0) else y
    return x, y

def get_data_train_tanh(url, width, height, depth, limit=1000, augmentation = False):
    
    files = glob.glob("{}/*.png".format(url))
    if("jpg" in url or "png" in url):
        files += glob.glob("{}".format(url))
    files += glob.glob("{}/*.jpg".format(url))
    files = files[:limit]
    x = []
    y = []
    for i, f in enumerate(files):
        if ((i+1) % 1000 == 0):
            print("[INFO] Download {} items finished!".format(i+1))
        img = cv2.imread(f, depth // 3)
        img = cv2.resize(img, (height, width))
        x.append(img)
        y.append(img.copy())
    if(augmentation):
        for i in range(len(x)):
            img = x[i]
            shape = img.shape
            for _ in range(10):
                ct = (int(np.random.rand(1) * shape[1]), int(np.random.rand(1) * shape[0]))
                img = cv2.rectangle(img, (ct[0]-2, ct[1]-2), (ct[0]+2, ct[1] + 2), (0,0,0), -1)
            x.append(img)
            y.append(y[i].copy())

    x = np.asarray(x)
    x = (x.astype("float32") / 127.5) - 1
    x = np.expand_dims(x, axis=-1) if(depth // 3 == 0) else x
    y = np.asarray(y)
    y = (y.astype("float32") / 127.5) - 1
    y = np.expand_dims(y, axis=-1) if(depth // 3 == 0) else y
    return x, y

if __name__ == "__main__":
    get_data(r"D:\Heal\Projects\NBB VI A13\Dataset\Screw\0",
             "./data/screw_a13/negative", 3000)
