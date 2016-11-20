import bisect
import random

from dataLoader import  *

def convert_img(img):
    """data is in RR..RG..GG..GB..BB format for each pixel"""

    new_image = np.empty((32,32,3),dtype=np.float16)

    for i in range(1024): #magic numbers runs in around 0.007-0.008sec
        new_image[i // 32][i % 32][0] = img[i] / 255
        new_image[i // 32][i % 32][1] = img[i+1024] / 255
        new_image[i // 32][i % 32][2] = img[i+2048] / 255

    return new_image

def calc_dist(a1, a2):
    return np.sum(np.abs(a2-a1))

def knn(training, test, k=7):
    """
    exact same function as in knn10
    """
    min_dists = []
    min_labels = []
    i = 0
    for item in training:
        i +=1

        dist = calc_dist(item[0], test)
        if len(min_dists) >= k: # TODO: optimise by doing first k items separately, then removing conditional
            q = bisect.bisect_left(min_dists, dist, lo=0, hi=k)
            if q == k:
                continue
            else:
                min_dists.insert(q, dist)
                min_dists.pop()
                min_labels.insert(q, item[1])
                min_labels.pop()
        else:
            q = bisect.bisect_left(min_dists, dist, lo=0, hi=len(min_dists))
            min_dists.insert(q, dist)
            min_labels.insert(q, item[1])

    x = [0 for i in range(100)] #magic number of classes
    for item in min_labels:
        x[item] += 1
    potential_classes = [k for k,i in enumerate(x) if i == max(x)]
    #print(potential_classes)
    return random.choice(potential_classes)



def knn100(test_data, blackwhite):
    """
    a simple knn data point to data point for each image, for CIFAR100
    """

    train_for_knn = []

    training = load_training_data100()
    for n, data in enumerate(training["data"]):
        data = (data-np.min(data))/(np.max(data)-np.min(data))
        label = training["fine_labels"][n]
        if blackwhite:
            data = convert_img(data)
            #data = normalise(data)
            data = np.dot(data[..., :3], [0.299, 0.587, 0.114])
        train_for_knn.append((data, label))


    guesses = []

    for n, data in enumerate(test_data["data"]):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        if blackwhite:
            data = convert_img(data)
            data = np.dot(data[..., :3], [0.299, 0.587, 0.114])

        guesses.append(knn(train_for_knn, data, 14))


    return guesses




if __name__=="__main__":
    import time

    test_data = load_test_batch100()

    _ = time.time()

    print(sum([i == test_data['fine_labels'][n] for n, i in enumerate(knn100(test_data,False))]))
    print(time.time() - _)
