from dataLoader import  *

from display import *

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


    colour_correct = 0
    results = np.zeros((100,100))
    for n, data in enumerate(test_data["data"]):
        #data = data / (np.max(data) - np.min(data))
        if not n%500:
            print(colour_correct, "/", n)

        total_class_guess = nearest_neighbour(Total_set, data)

        correct_class = test_data["fine_labels"][n]
        results[total_class_guess][correct_class] += 1
        if total_class_guess == correct_class:
            colour_correct += 1

    save_image("averager100fine-conf-matrix.png", results, (100,100), (100,100))

    print("All colours:",colour_correct)




if __name__=="__main__":
    test_data = load_test_batch100()
    averager100fine(test_data)