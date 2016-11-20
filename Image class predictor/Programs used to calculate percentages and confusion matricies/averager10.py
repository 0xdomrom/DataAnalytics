from dataLoader import  *

from display import *

def calc_dist(a1, a2):
    #return np.linalg.norm(a2 - a1)
    return np.sum(np.abs(a2-a1))


def nearest_neighbour(training_set, test):
    minimal = 307300 # as differences are floats/ <1
    result_class = -1
    for n, item in enumerate(training_set):

        dist = calc_dist(item, test)
        if dist < minimal:

            minimal = dist
            result_class = n

    return result_class

def averager10(test_data):
    averages = {i:np.zeros((3072,),dtype=np.float64)for i in range(10)}
    count = {i:0 for i in range(10)}
    example_img = {i:None for i in range(10)}
    for training_set in range(1,6):
        training = load_training_data10(training_set)
        for n, data in enumerate(training["data"]):
            data = (data-np.min(data))/(np.max(data)-np.min(data))
            label = training["labels"][n]
            averages[label] += data
            count[label] += 1
            example_img[label] = data

    Total_set = []
    Gray_set = []
    #plot = MultiImage()
    for i in averages:
        img = averages[i]/count[i]
        Total_set.append(img)
        Gray_set.append(to_grayscale(convert_img(img)))
    #    plot.add(convert_img(Total_set[-1]))
        # #plot.add(Gray_set[-1],"gray")
    #    plot.add(convert_img(example_img[i]))

    #plot.draw()

    colour_correct = 0
    grey_correct = 0

    colour_results = np.zeros((10,10))
    grey_results = np.zeros((10,10))


    for n, data in enumerate(test_data["data"]):
        data = data / (np.max(data) - np.min(data))
        if not n%1000:
            print(n)
            # print(colour_correct, "/", n)
            # print(grey_correct, "/", n)

        data_gray = to_grayscale(convert_img(data))

        colour_class_guess = nearest_neighbour(Total_set, data)

        grey_class_guess = nearest_neighbour(Gray_set, data_gray)

        correct_class = test_data["labels"][n]

        colour_results[colour_class_guess][test_data["labels"][n]] += 1
        grey_results[grey_class_guess][test_data["labels"][n]] += 1



        if colour_class_guess == correct_class:
            colour_correct += 1

        if grey_class_guess == correct_class:
            grey_correct += 1

    save_image("averager10-colour-normalised-conf-matrix.png", colour_results, (10,10), (10,10))
    save_image("averager10-grey-normalised-conf-matrix.png", grey_results, (10,10), (10,10))


    print("All colours:",colour_correct, "/", 10000)
    print("Gray:",grey_correct, "/", 10000)



if __name__=="__main__":

    test_data = load_test_batch10()

    averager10(test_data)