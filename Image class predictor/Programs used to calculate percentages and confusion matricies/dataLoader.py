import pickle
import time

import numpy as np
from display import *
import scipy.misc as spm


def load_training_data10(batch):
    with open("/Users/Dom/Desktop/Data Analytics/cifar-10-batches-py/data_batch_"+str(batch), "rb") as f:
        x = pickle.load(f, encoding="latin1")
    print("CIFAR10 training batch "+str(batch) + " loaded")
    return x

def load_test_batch10():
    with open("/Users/Dom/Desktop/Data Analytics/cifar-10-batches-py/test_batch", "rb") as f:
        x = pickle.load(f, encoding="latin1")
    print("CIFAR10 test batch loaded")
    return x

def load_training_data100():
    with open("/Users/Dom/Desktop/Data Analytics/cifar-100-python/train", "rb") as f:
        x = pickle.load(f, encoding="latin1")
    print("CIFAR100 training data")
    return x

def load_test_batch100():
    with open("/Users/Dom/Desktop/Data Analytics/cifar-100-python/test", "rb") as f:
        x = pickle.load(f, encoding="latin1")
    print("CIFAR 100 test data")
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

    for item in data["data"]:
        img = data["data"][0]

        # _ = time.time()
        # # TODO: if you can make it a 1D np array of RGBRGBRGB, faster than 1/100th a sec, this will do the rest super fast
        # reshape = img.reshape(32,32,3)
        # # if theres a way to make this work, kudos to you
        # print("time to reshape:", str(time.time()-_))

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
