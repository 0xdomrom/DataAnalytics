import averager10, averagertwo10, averagertwo100, averager100fine, knn10, knn100, averager100coarse_fine
from dataLoader import *


def reverse_convert(data):
    x = np.zeros((3072,),dtype=np.uint8)
    for n,row in enumerate(data):
        for m,item in enumerate(row):
            for l,i in enumerate(item):
                x[l*1024+n*32+m] = i
    return x




if __name__=="__main__":
    """
    to run a specific classifier, to calculate a percentage correct
    """

    test_data = {"data":[]}

    i = 1
    print("img"+["0"+str(i), str(i)][i>9])

    while True:
        try:
            img = spm.imread("img"+["0"+str(i), str(i)][i>9])
            test_data["data"].append(reverse_convert(img))
            i += 1
        except:
            break

    results = []
    if True:
        print("Averager CIFAR10")
        results.append(averager10.averager10(test_data))
        print("Averager colour permutations CIFAR10")
        results.append(averagertwo10.averagertwo10(test_data))
        print("KNN CIFAR10")
        results.append(knn10.knn10(test_data,False))
        print("Averager CIFAR100 Fine labels")
        results.append(averager100fine.averager100fine(test_data))
        print("Averager CIFAR100 Coarse then Fine labels")
        results.append(averager100coarse_fine.averager100coarse_fine(test_data))
        print("Averager colour permutations CIFAR100")
        results.append(averagertwo100.averagertwo100(test_data))
        print("KNN CIFAR100")
        results.append(knn100.knn100(test_data,False))
        print(results)

    import csv

    with open('output.csv', 'wb') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerows(results)
