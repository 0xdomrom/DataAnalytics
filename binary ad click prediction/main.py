print "=program starting="

from pyspark import SparkContext
from Hasher import * #has np


sc = SparkContext("local", "Simple App")

def clean_data(point):
    point = [i if i else -1 for i in point.split(",")]
    try:
        return np.asarray(point[:14], dtype=np.int32)
    except:
        print(point[:14])
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
        print 'An example of rawData entry: ', rawData.takeSample(False, 1)
        print 'An example of rawData entry: ', rawData.take(1)

        nData = rawData.count()
        print '\nrawData count=', nData
        return rawData
    else:
        print 'file not found', fileName
        return None


def split_data(rawData):
    weights = [.9, .1]
    seed = 17
    # Use randomSplit with weights and seed
    rawTrainData, rawTestData = rawData.randomSplit(weights, seed)
    # Cache the data
    # this will make accessing it much faster later when doing KNN
    rawTrainData.cache()
    rawTestData.cache()

    nTrain = rawTrainData.count()
    nTest = rawTestData.count()
    print 'rawTrainData count=', nTrain
    print 'rawTestData count=', nTest

    return rawTrainData, rawTestData

def test_only_no(test_data):
    from operator import add
    # .collect() makes it a python list
    count_zero = test_data.map(lambda x:0 if int(x[0]) else 1).collect()
    print sum(count_zero), "/", test_data.count()

def addSeeds(LSHhash):
    def func_wrapper(data):
        return LSHhash(data, 1,2)
    return func_wrapper

@addSeeds
def LSHhash(data, s1, s2):
    hash = 5461411907
    hash = (hash * 17126785) ^ s1
    hash = (hash * 17126785) ^ s2
    hash = (hash * 17126785) ^ data
    return np.sum(hash) % 1e12


def parse_point(point,n=None):
    """converts [label, feature1...featuren] => [(label,feature1),...,(label,featuren)]"""
    point = point.split(",")
    if not n:
        n = len(point)
    return [(point[0],item) for item in point[1:n] if item]

def one_hot_encoding(train_data):
    parsedTrainFeat = train_data.map(parse_point)
    print 'parsedTrainFeat: ', parsedTrainFeat.take(1) #display one
    numCategories = (parsedTrainFeat
        .flatMap(lambda x: x)
        .distinct()
        .map(lambda x: (x[0], 1))
        .reduceByKey(lambda x, y: x + y)
        .sortByKey()
        .collect())

    print '\nnumCategories=', numCategories



if __name__=="__main__":
    raw_data = load_data("dac_sample.txt")
    if not raw_data:
        raise Exception("FATAL: no sample data found")
    training_data, test_data = split_data(raw_data)
    #test_only_no(test_data)
    #one_hot_encoding(training_data)
    hashRows(training_data)





