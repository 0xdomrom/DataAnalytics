import bisect
import numpy as np
import random
import time



def knn(training, test, k, distance_metric):
    """
    Memory considerate knn algorithm
    :param training: training data
    :param test: point to test
    :param k: amount of closest to compare
    :return:
    """
    min_dists = []
    min_labels = []
    i = 0

    for i in range(k):
        item = training[i]
        dist = distance_metric(item[1], test)

        q = bisect.bisect_left(min_dists, dist, lo=0, hi=len(min_dists))
        min_dists.insert(q, dist)
        min_labels.insert(q, item[0])

    lowest = min_dists[-1] # TODO: is checking lowest before bisect faster?

    for item in training[k:]:
        i +=1

        dist = distance_metric(item[1], test)
        if dist < lowest:
            q = bisect.bisect_left(min_dists, dist, lo=0, hi=k)
            # bisect left finds the first item in a sorted array that has a value greater than dist
            if q == k:
                    #if value is beyond list, skip it
                continue
            else:
                # otherwise insert into correct sorted place, and pop last element, to keep k length
                min_dists.insert(q, dist)
                min_dists.pop()
                min_labels.insert(q, item[0])
                min_labels.pop()
                lowest = min_dists[-1]

    if sum(min_labels)*2 == k:
        return random.randint(0,1)
    return int(sum(min_labels)*2 > k)


if __name__=="__main__":
    def run():
        lol()
        return knn(q, np.asarray([random.randint(0, 5000) for i in range(30)]), 7)

    z = []
    results = []
    for i in range(100):
        print i
        _ = time.time()
        results.append(run())
        z.append(time.time()-_)
    print sum(results)
    print sum(z)/len(z)
    print "min:", min(z)
    print "max:", max(z)

    # 0.267101845741 with if dist < lowest
    # 0.269304838181 without if
    # 0.300723376274
    # min: 0.252966165543
    # max: 0.326858997345
