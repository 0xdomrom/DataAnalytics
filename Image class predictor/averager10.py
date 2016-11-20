from dataLoader import  *

def convert_img(img):
    """data is in RR..RG..GG..GB..BB format for each pixel"""

    new_image = np.empty((32,32,3),dtype=np.float16)

    for i in range(1024): #magic numbers runs in around 0.007-0.008sec
        new_image[i // 32][i % 32][0] = img[i] / 255
        new_image[i // 32][i % 32][1] = img[i+1024] / 255
        new_image[i // 32][i % 32][2] = img[i+2048] / 255

    #new_image = np.array([[img[i],img[i+1024],img[i+2048]] for i in range(1024)],)


    return new_image
def calc_dist(a1, a2):
    #return np.linalg.norm(a2 - a1)
    return np.sum(np.abs(a2-a1))


def nearest_neighbour(training_set, test):
    minimal = 783360 # as differences should be less than 255*3072
    result_class = -1
    for n, item in enumerate(training_set):

        dist = calc_dist(item, test)
        if dist < minimal:

            minimal = dist
            result_class = n

    return result_class

def averager10(test_data):
    """
    add each datapoint to each related pixel, for each label, to result in 10 average images
    which are then compared to by a 1nn algorithm
    """
    averages = {i:np.zeros((3072,),dtype=np.float64)for i in range(10)}
    count = {i:0 for i in range(10)}
    example_img = {i:None for i in range(10)}
    for training_set in range(1,6):
        training = load_training_data10(training_set)
        for n, data in enumerate(training["data"]):
            #data = (data-np.min(data))/(np.max(data)-np.min(data))
            label = training["labels"][n]
            averages[label] += data
            count[label] += 1
            example_img[label] = data

    Total_set = []
    for i in averages:
        img = averages[i]/count[i]
        Total_set.append(img)

    guesses = []


    for n, data in enumerate(test_data["data"]):
        #data = data / (np.max(data) - np.min(data))

        colour_class_guess = nearest_neighbour(Total_set, data)
        guesses.append(colour_class_guess)

    return guesses




if __name__=="__main__":
    test_data = load_test_batch10()
    print(sum([i==test_data['labels'][n] for n, i in enumerate(averager10(test_data))]))