import numpy as np
from dataLoader import  *
from display import *


def calc_dist(a1, a2):
    # return np.linalg.norm(a2 - a1)
    # if a1.shape == (3072,):
    #     print(a1)
    return np.sum(np.abs(a2-a1))


def nearest_neighbour(training_set, test):
    minimal = 3073 # as differences are floats/ <1
    result_class = -1
    for label in training_set:
        for item in training_set[label]:
            dist = calc_dist(item, test)
            if dist < minimal:
                minimal = dist
                result_class = label

    return result_class

def averagertwo100(test_data):
    averages = {i:[np.zeros((3072,),dtype=np.float64)]for i in range(100)}
    count = {i:0 for i in range(100)}
    max_vals = {i:[0,0,0] for i in range(100)}

    training = load_training_data100()
    for n, data in enumerate(training["data"]):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        label = training["fine_labels"][n]
        averages[label][-1] += data
        count[label] += 1
        if np.sum(data[:1024]) > max_vals[label][0]:
            max_vals[label][0] = np.sum(data[:1024])
        if np.sum(data[1024:2048]) > max_vals[label][1]:
            max_vals[label][1] = np.sum(data[1024:2048])
        if np.sum(data[2048:]) > max_vals[label][2]:
            max_vals[label][2] = np.sum(data[2048:])

    for item in averages:
        averages[item][0] = convert_img(averages[item][0]/count[item])

    amounts = {_:{k:{j:{i:0 for i in range(3)} for j in range(3)} for k in range(3)} for _ in range(100)}
    a = {_:{k:{j:{i:np.zeros((3072,)) for i in range(3)} for j in range(3)} for k in range(3)} for _ in range(100)}

    training = load_training_data100()
    for n, data in enumerate(training["data"]):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        label = training["fine_labels"][n]

        if np.sum(data[:1024]) > 2*max_vals[label][0]/3:
            k=2
        elif np.sum(data[:1024]) > max_vals[label][0]/3:
            k=1
        else:
            k=0

        if np.sum(data[1024:2048]) > 2*max_vals[label][1]/3:
            j=2
        elif np.sum(data[1024:2048]) > max_vals[label][1]/3:
            j=1
        else:
            j=0

        if np.sum(data[2048:]) > 2*max_vals[label][2]/3:
            i=2
        elif np.sum(data[2048:]) > max_vals[label][2]/3:
            i=1
        else:
            i=0
        amounts[label][k][j][i] += 1
        a[label][k][j][i] += data



    for label in range(10):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if amounts[label][k][j][i]:

                        data = a[label][k][j][i] / amounts[label][k][j][i]

                        averages[label].append(convert_img(data))

    #
    # for label in averages:
    #     plot = MultiImage()
    #     for item in averages[label]:
    #         plot.add(item)
    #     plot.draw()

    results = np.zeros((100,100))
    correct = 0
    for n, data in enumerate(test_data["data"]):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        if not n%500:
            print(correct, "/", n)

        data = convert_img(data)

        guessed_class = nearest_neighbour(averages, data)

        correct_class = test_data["fine_labels"][n]
        results[guessed_class][correct_class] += 1
        if guessed_class == correct_class:
            correct += 1
    save_image("averagertwo100-conf-matrix.png", results, (100,100), (100,100))


    print("Taxicab:",correct, "/", 10000)




if __name__=="__main__":
    test_data = load_test_batch100()
    averagertwo100(test_data)