from dataLoader import  *

from display import *

def calc_dist(a1, a2):
    #return np.linalg.norm(a2 - a1)
    return np.sum(np.abs(a2-a1))


def nearest_neighbour(training_set, test):
    minimal = 500000 # as differences are floats/ <1
    result_class = -1
    for key in training_set:
        dist = calc_dist(training_set[key], test)
        if dist < minimal:
            minimal = dist
            result_class = key

    return result_class

def averager100coarse_fine(test_data):
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


    coarse_correct = 0
    fine_correct = 0
    results = np.zeros((100,100))
    for n, data in enumerate(test_data["data"]):
        data = (data-np.min(data))/(np.max(data)-np.min(data))
        if not n%1000:
            print(n)

        q = nearest_neighbour(coarse_total, data)
        if q != test_data["coarse_labels"][n]:
            # save time of computation if coarse label incorrect
            continue
        coarse_correct += 1

        fin = nearest_neighbour(fine_total[q],data)
        results[fin][test_data["fine_labels"][n]] += 1

        if fin == test_data["fine_labels"][n]:
            fine_correct += 1
    save_image("averager100coarse-fine-conf-matrix.png", results, (100,100), (100,100))


    print("Coarse correct:",coarse_correct, "/ 10000")
    print("Fine correct:",fine_correct, "/ 10000")



if __name__=="__main__":
    test_data = load_test_batch100()
    averager100coarse_fine(test_data)