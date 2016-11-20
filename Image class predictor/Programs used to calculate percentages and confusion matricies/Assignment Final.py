import averager10, averagertwo10, averagertwo100, averager100fine, knn10, knn100, averager100coarse_fine
from dataLoader import *
from display import *
import multiprocessing




if __name__=="__main__":
    """
    to run a specific classifier, to calculate a percentage correct, uncomment a line
    """
    test_data_10 = load_test_batch10()
    test_data_100 = load_test_batch100()

    #
    # averager10.averager10(test_data_10)
    # averagertwo10.averagertwo10(test_data_10)
    # knn10.knn10(test_data_10,False)
    # averager100fine.averager100fine(test_data_10)
    # averager100coarse_fine.averager100coarse_fine(test_data_10)
    # averagertwo100.averagertwo100(test_data_10)
    # knn100.knn100(test_data_10,False)
    #
