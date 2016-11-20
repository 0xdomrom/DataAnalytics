"""
dataLoader.py

for importing data, and converting it to a format useable by image displays
"""

import pickle
import time

import numpy as np
import scipy.misc as spm


def load_training_data10(batch):
    with open("/Users/Dom/Desktop/Data Analytics/cifar-10-batches-py/data_batch_"+str(batch), "rb") as f:
        x = pickle.load(f, encoding="latin1")
    return x

def load_test_batch10():
    with open("/Users/Dom/Desktop/Data Analytics/cifar-10-batches-py/test_batch", "rb") as f:
        x = pickle.load(f, encoding="latin1")
    return x

def load_training_data100():
    with open("/Users/Dom/Desktop/Data Analytics/cifar-100-python/train", "rb") as f:
        x = pickle.load(f, encoding="latin1")
    return x

def load_test_batch100():
    with open("/Users/Dom/Desktop/Data Analytics/cifar-100-python/test", "rb") as f:
        x = pickle.load(f, encoding="latin1")
    return x

def to_grayscale(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

def red_channel(img):
    return img[..., 0]

def green_channel(img):
    return img[..., 1]

def blue_channel(img):
    return img[..., 2]


def save_image(name, image, in_shape=(3,32,32), out_dim=(512,512)):
    img = spm.toimage(image.reshape(in_shape)).resize(out_dim)
    spm.imsave(name, img)




if __name__ == "__main__":
    _ = time.time()

    data = load_training_data10(1)

    print("time to load dataset:",time.time()-_)


    def convert_img(img):
        """data is in RR..RG..GG..GB..BB format for each pixel"""

        new_image = np.empty((32, 32, 3), dtype=np.float16)

        for i in range(1024):  # magic numbers runs in around 0.007-0.008sec
            new_image[i // 32][i % 32][0] = img[i] / 255
            new_image[i // 32][i % 32][1] = img[i + 1024] / 255
            new_image[i // 32][i % 32][2] = img[i + 2048] / 255
        return new_image

    for item in data["data"]:
        img = data["data"][0]


        _ = time.time()
        img = convert_img(img)
        print("time to convert:", time.time()-_)

        _ = time.time()
        grayscale = to_grayscale(img)
        print("time to gray:", time.time()-_)

        _ = time.time()
        red = red_channel(img)
        green = green_channel(img)
        blue = blue_channel(img)
        print("time to get separate colours:", time.time()-_)

        break
    test = load_test_batch100()
    print(test.keys())
    data = load_training_data100()
