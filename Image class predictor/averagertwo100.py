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
    """
    worse and slower than averager100, but worth testing (same as averagertwo10, but with 100 fine_labels as classes)
    Didn't attempt coarse than fine labels using this, but tried with averager100, which had much worse results, like 3%
    """
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
        averages[item][0] = averages[item][0]/count[item]

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

                        averages[label].append(data)

    #
    # for label in averages:
    #     plot = MultiImage()
    #     for item in averages[label]:
    #         plot.add(item)
    #     plot.draw()

    guesses = []
    for n, data in enumerate(test_data["data"]):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        guesses.append(nearest_neighbour(averages, data))

    return guesses


if __name__=="__main__":
    import time

    test_data = load_test_batch100()

    _ = time.time()

    print(sum([i == test_data['fine_labels'][n] for n, i in enumerate(averagertwo100(test_data))]))
    print(time.time() - _)
