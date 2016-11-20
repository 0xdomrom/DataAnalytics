import bisect
import random

from dataLoader import  *
from display import *


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

    train_for_knn = []

    training = load_training_data100()
    for n, data in enumerate(training["data"]):
        data = (data-np.min(data))/(np.max(data)-np.min(data))
        if not n%10000:
            print(n)
        label = training["fine_labels"][n]
        if blackwhite:
            data = convert_img(data)
            #data = normalise(data)
            data = np.dot(data[..., :3], [0.299, 0.587, 0.114])
        train_for_knn.append((data, label))
    print("loaded training")


    correct = 0
    results = np.zeros((100,100))


    for n, data in enumerate(test_data["data"]):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        if blackwhite:
            data = convert_img(data)
            data = np.dot(data[..., :3], [0.299, 0.587, 0.114])

        knn_class_guess = knn(train_for_knn, data, 14)


        results[knn_class_guess][test_data["fine_labels"][n]] += 1

        if knn_class_guess == test_data["fine_labels"][n]:
            correct += 1
            print(correct, "/", n + 1)
    print(correct, "/", n+1)
    save_image("knn100-all10000.png", results, (100,100), (100,100))




if __name__=="__main__":
    test_data = load_test_batch100()

    knn100(test_data, False)
    #do_knn([1,2,3,4,5],True)
    pass
