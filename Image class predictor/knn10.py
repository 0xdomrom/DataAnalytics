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
    # TODO: np.sum(np.square(diff)) or np.dot(diff, diff) faster?
    #return np.sum(np.square(item - test))
    #return np.linalg.norm(a2-a1)
    return np.sum(np.abs(a2-a1))

def knn(training, test, k=7):
    """
    KNN sort for test item, using training set training


    TODO:   make generator getting items from training
            get first k items before doing main loop over the rest of the generator
            to cut out if  min_dists > k


    """
    min_dists = []
    min_labels = []
    i = 0
    for item in training:
        i +=1

        dist = calc_dist(item[0], test)
        if len(min_dists) >= k:
            q = bisect.bisect_left(min_dists, dist, lo=0, hi=k)
            # bisect left finds the first item in a sorted array that has a value greater than dist
            if q == k:
                #if value is beyond list, skip it
                continue
            else:
                # otherwise insert into correct sorted place, and pop last element, to keep k length
                min_dists.insert(q, dist)
                min_dists.pop()
                min_labels.insert(q, item[1])
                min_labels.pop()
        else:
            q = bisect.bisect_left(min_dists, dist, lo=0, hi=len(min_dists))
            min_dists.insert(q, dist)
            min_labels.insert(q, item[1])

    x = [0 for i in range(10)] #magic number of classes
    for item in min_labels:
        x[item] += 1
    potential_classes = [k for k,i in enumerate(x) if i == max(x)]
    return random.choice(potential_classes)



def knn10(test_data, blackwhite):
    """
    a simple knn data point to data point for each image, for CIFAR10
    optional blackwhite boolean (slows program down considerably, for not much of an increace, if any at all

    as convert_img is fairly slow, faster to do colour, however results for greyscale could be better, not fully tested
    """
    train_for_knn = []
    for i in range(1,6):
        training = load_training_data10(i)
        for n, data in enumerate(training["data"]):
            data = (data-np.min(data))/(np.max(data)-np.min(data))

            label = training["labels"][n]
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
            # data = normalise(data)
            data = np.dot(data[..., :3], [0.299, 0.587, 0.114])

        guesses.append(knn(train_for_knn, data, 14))


    return guesses

if __name__=="__main__":
    import time

    test_data = load_test_batch100()

    _ = time.time()

    print(sum([i == test_data['fine_labels'][n] for n, i in enumerate(knn10(test_data,False))]))
    print(time.time() - _)