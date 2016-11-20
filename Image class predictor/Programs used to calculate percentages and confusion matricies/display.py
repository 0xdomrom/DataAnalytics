
import matplotlib.pyplot as pl
import numpy as np


class MultiImage:
    def __init__(self):
        self.size = 0
        self.plots = []
    def add(self, img, cmap=None):
        self.size += 1
        self.plots.append((img,cmap))

    def draw(self):

        if self.size > 3:
            height = np.ceil(np.sqrt(self.size))
            width = np.ceil(self.size/height)
            height = int(height)
            width = int(width)
        else:
            height = 1
            width = self.size

        fig = pl.figure()
        for i in range(self.size):
            a = fig.add_subplot(height,width,i+1)
            if self.plots[i][1]:
                pl.imshow(self.plots[i][0], cmap=self.plots[i][1])
            else:
                pl.imshow(self.plots[i][0])
            a.set_title(i+1)
        pl.show()








def convert_img(img):
    """data is in RR..RG..GG..GB..BB format for each pixel"""

    new_image = np.empty((32,32,3),dtype=np.float16)

    for i in range(1024): #magic numbers runs in around 0.007-0.008sec
        new_image[i // 32][i % 32][0] = img[i] / 255
        new_image[i // 32][i % 32][1] = img[i+1024] / 255
        new_image[i // 32][i % 32][2] = img[i+2048] / 255

    #
    # # change data to RGB,RGB,...,RGB
    # # variable length solution, runs in about 0.010 - 0.012sec
    # for i in range(len(img)):
    #     if i < len(img) // 3:
    #         new_image[i // 32][i % 32][0] = img[i] /255
    #     elif len(img) // 3 < i < 2 * len(img) // 3:
    #         new_image[(i - len(img) // 3) // 32][i % 32][1] = img[i]/255
    #     else:
    #         new_image[(i - 2 * len(img) // 3) // 32][i % 32][2] = img[i]/255

    #new_image = np.array([[img[i],img[i+1024],img[i+2048]] for i in range(1024)],)


    return new_image