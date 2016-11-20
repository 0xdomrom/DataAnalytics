from pyspark import SparkContext
import numpy as np
import random
import bisect

c1 = 0
c2 = 0

def calc_dist(point1, point2):
    #
    # tot = 0
    # for i in xrange(len(point1)):
    #     if point1[i] == -1 or point2[i] == -1:
    #         continue
    #     tot += np.square(point1[i]-point2[i])
    # return np.sqrt(tot)
    return np.linalg.norm(point1-point2)



def clean_data(point):
    # H: cut off text based features
    point = [i if i else -1 for i in point.split(",")[:14]]
    for i in [12, 10, 1]:
        point.pop(i)

    try:
        #return map(int, point)
        return np.asarray(point, dtype=np.float32)
    except:
        print(point)
        raise


def load_data(file_location):
    import os.path
    fileName = os.path.join(file_location)

    partitions = 1
    if os.path.isfile(fileName):
        rawData = (sc
                   .textFile(fileName, partitions)
                   .map(lambda x: x.replace('\t',','))
                   .map(clean_data))  # work with either ',' or '\t' separated data
        max_vals = rawData.fold(np.zeros(11),lambda x, y:[max(i,y[n]) for n, i in enumerate(x)])
        rawData = rawData.map(lambda x:x/max_vals)

        return rawData
    else:
        print '==file not found', fileName
        return None


def split_data(rawData, seed=17, test_missing_data=False):
    weights = [.9, .1]
    rawTrainData, rawTestData = rawData.randomSplit(weights, seed)
    #rawTrainData = (rawTrainData
    #    .filter(lambda x: False if x.count(-1)>3 else True))
    print rawTrainData.count()
    if test_missing_data:
        for i in range(11):
            temp = (rawTrainData
                    .filter(lambda x: bool(x[i]+1)))# as if missing data is -1,
                                                    # bool(0) = False, otherwise True
            print str(i)+ "\t"+str(temp.count())
    rawTrainData = rawTrainData.filter(lambda x: -1 not in x)
    # only use data that
    print "==training data with no features missing", rawTrainData.count()
    rawTrainData.cache()
    rawTestData.cache()

    return rawTrainData, rawTestData





class ClustersObject:
    def __init__(self, data, k):
        self.k = k
        self.centroids = data.takeSample(False, k)
        self.clusters = []

    @staticmethod
    def closest_centroid(this):
        centroids = this.centroids
        def _closest_centroid(test_point):
            lowest = 0
            lowest_dist = 0
            for n, i in enumerate(centroids):
                dist = calc_dist(i[1:], test_point[1][1:])

                if n == 0:
                    lowest_dist = dist
                elif dist < lowest_dist:
                    lowest_dist = dist
                    lowest = n
            return [lowest, test_point[1]]

        return _closest_centroid



    def train(self, data, runs=1, e=0.0001):

        for i in xrange(runs): # H: xrange not range py2 el mao zedong
            new = []

            print "==Training run:", i+1
            # TODO:mapreduce -> sc.parallelize(c, partitions).map(...)
            data = data.map(
                ClustersObject.closest_centroid(self)
            )
            data.cache()
            for centroid_num in range(self.k):
                c = data.filter(lambda x:x[0]==centroid_num)

                c_len = c.count()
                print "===centroid", centroid_num, "has", c_len, "around it"

                if c_len > 0:
                    # TODO:mapreduce -> sc.parallelize(c, partitions).fold(...)
                    end = c.map(
                        lambda x: x[1])\
                        .fold(np.zeros(11),(lambda x, y:x+y))

                    new.append(end/c_len)
                else:
                    print "==no points around centroid", centroid_num
                    raise Exception

            self.centroids = new
        print "==clusters compiling"
        self.clusters = [ # H: hahahaha memory
            data.filter(lambda x:x[0]==i).map(lambda x:x[1]).collect() for i in range(self.k)
        ]
        return data, self.clusters


def knn_wrap(k, distance_metric, clusters):
    def knn(test_item):
        global c1, c2
        test = test_item[1:] # H: probs this
        min_cluster_dist = 0

        min_cluster_index = 0

        for i in range(k):
            item = clusters[i]
            dist = distance_metric(item[1][1:], test)
            if i == 0:
                min_cluster_dist = dist
                min_cluster_index = i
            else:
                if dist < min_cluster_dist:
                    min_cluster_dist = dist
                    min_cluster_index = i

        training = clusters[min_cluster_index]

        min_dists = []
        min_labels = []
        i = 0

        for i in range(k):
            item = training[i]
            dist = distance_metric(item[1:], test)

            q = bisect.bisect_left(min_dists, dist, lo=0, hi=len(min_dists))
            min_dists.insert(q, dist)
            min_labels.insert(q, item[0])

        lowest = min_dists[-1]  # TODO: is checking lowest before bisect faster?

        for item in training[k:]:
            i += 1

            dist = distance_metric(item[1:], test)
            if dist < lowest:
                q = bisect.bisect_left(min_dists, dist, lo=0, hi=k)
                # bisect left finds the first item in a sorted array that has a value greater than dist
                if q == k:
                    # if value is beyond list, skip it
                    continue
                else:
                    # otherwise insert into correct sorted place, and pop last element, to keep k length
                    min_dists.insert(q, dist)
                    min_dists.pop()
                    min_labels.insert(q, item[0])
                    min_labels.pop()
                    lowest = min_dists[-1]

        if sum(min_labels) * 2 == k:
            return 0
        c1 += int(sum(min_labels) * 2 > k) == test_item[0]
        c2 += 1
        print c1, c2
        return int(sum(min_labels) * 2 > k)
    return knn



if __name__=='__main__':

    print "\n\n=program starting"

    sc = SparkContext("local", "Simple App")
    print "=loading data"
    raw_data = load_data("././dac_sample.txt")
    print "=splitting data"
    train_data, test_data = split_data(raw_data)

    print "=clustering"
    clusters = ClustersObject(train_data, 12)

    train_data = train_data.map(lambda x:[-1, x])

    train_data, clusterarray = clusters.train(train_data)

    train_data.cache()
    print "=knn time"
    results = test_data.map(knn_wrap(7, calc_dist, clusterarray)).collect()
    print "=count correct time"
    test_data_labels = test_data.map(lambda x: x[0]).collect()
    correct = 0
    counted = 0
    for n, item in enumerate(results):
        counted += 1
        if item == test_data_labels[n]:
            correct += 1

        if n%100==99:
            print correct, "/", counted

    print correct, counted

    print "\n=program end"



"""
Determine whether user clicked add:
- KNN using LSH to improve computation & use map reduce
- Assignment goal is to implement this
"""