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
    #return np.linalg.norm(a2 - a1)
    return np.sum(np.abs(a2-a1))


def nearest_neighbour(training_set, test):
    minimal = 500000 # as differences are floats/ <1
    result_class = -1
    for n, item in enumerate(training_set):
        dist = calc_dist(item, test)
        if dist < minimal:
            minimal = dist
            result_class = n

    return result_class

def averager100fine(test_data):
    averages = {i:np.zeros((3072,),dtype=np.float64)for i in range(100)}
    count = {i:0 for i in range(100)}

    training = load_training_data100()
    for n, data in enumerate(training["data"]):
        #data = (data-np.min(data))/(np.max(data)-np.min(data))
        label = training["fine_labels"][n]
        averages[label] += data
        count[label] += 1

    Total_set = []

    for i in averages:
        img = averages[i]/count[i]
        Total_set.append(img)

    guesses = []
    for n, data in enumerate(test_data["data"]):
        #data = data / (np.max(data) - np.min(data))

        guesses.append(nearest_neighbour(Total_set, data))

    return guesses



if __name__=="__main__":
    import time
    test_data = load_test_batch100()
    _ = time.time()
    print(sum([i==test_data['fine_labels'][n] for n, i in enumerate(averager100fine(test_data))]))
    print(time.time()-_)