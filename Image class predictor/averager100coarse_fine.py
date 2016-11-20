from dataLoader import  *

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
def calc_dist(a1, a2):
    #return np.linalg.norm(a2 - a1)
    return np.sum(np.abs(a2-a1))


def nearest_neighbour(training_set, test):
    minimal = 783360 # magic number
    result_class = -1
    for key in training_set:
        dist = calc_dist(training_set[key], test)
        if dist < minimal:
            minimal = dist
            result_class = key

    return result_class

def averager100coarse_fine(test_data):
    """
    generates average images for each superclass, as well as each class
    Then guesses which superclass the test data belongs to, and then the subclass
    """
    coarse_averages = {i:np.zeros((3072,),dtype=np.float64)for i in range(20)}
    coarse_count = {i:0 for i in range(20)}
    fine_averages = {i:{}for i in range(20)}
    fine_count = {i:{} for i in range(20)}

    training = load_training_data100()
    for n, data in enumerate(training["data"]):
        data = (data-np.min(data))/(np.max(data)-np.min(data))
        coarse_label = training["coarse_labels"][n]
        coarse_averages[coarse_label] += data
        coarse_count[coarse_label] += 1

        fine_label = training["fine_labels"][n]

        if fine_label in fine_averages[coarse_label]:
            fine_averages[coarse_label][fine_label] += data
            fine_count[coarse_label][fine_label] += 1
        else:
            fine_averages[coarse_label][fine_label] = data
            fine_count[coarse_label][fine_label] = 1

    coarse_total = {i:None for i in range(20)}
    fine_total = {i:{} for i in range(20)}

    for i in coarse_averages:
        coarse_total[i] = coarse_averages[i]/coarse_count[i]
        for j in fine_averages[i]:
            fine_total[i][j] = fine_averages[i][j]/fine_count[i][j]

    guesses = []

    for n, data in enumerate(test_data["data"]):
        data = (data-np.min(data))/(np.max(data)-np.min(data))

        q = nearest_neighbour(coarse_total, data)

        fin = nearest_neighbour(fine_total[q],data)

        guesses.append(fin)
    return guesses


if __name__=="__main__":
    import time

    test_data = load_test_batch100()
    _ = time.time()
    print(sum([i == test_data['fine_labels'][n] for n, i in enumerate(averager100coarse_fine(test_data))]))
    print(time.time() - _)