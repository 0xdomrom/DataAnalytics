import time

from dataLoader import *


dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]





def crappy_edge_detect(img,epsilon=0.1):
    """
    assumes img 32x32 or bigger
    """
    Result = [[False for i in range(32)] for i in range(32)]

    for y in range(1,len(img)-1):
        if not y%50:
            print(y)
        for x in range(1,31):
            z = {i:img[y+i[0]][x+i[1]] for i in dirs}
            current = img[y][x]
            if 3 < sum([abs(current-z[dir])<epsilon for dir in z]) < 6:
                Result[y][x] = True

    return Result
    # return [[float(i) for i in j] for j in Result]


def knn(training, labels, k=7):
    test_data = load_test_batch10()
    count_correct = 0


    for j in range(len(test_data["data"])):
        _ = time.time()
        distances = [0 for i in range(len(training))]
        data = convert_img(test_data["data"][j])
        label = test_data["labels"][j]

        img = [[i[0] * 0.299 + i[1] * 0.587 + i[2] * 0.114 for i in j] for j in data]
        img = crappy_edge_detect(img)

        for n, row in enumerate(img):
            for m, x in enumerate(row):
                for i,item in enumerate(training):
                    if x != item[n][m]:
                        distances[i] += 1

        minimals = []
        count = 0
        for i in range(1024):
            minimals.append([])
            for n, item in enumerate(distances):
                if item == i:
                    minimals[-1].append(labels[n])
                    count +=1
            if count >= k:
                break

        final = []
        for item in minimals:
            for i in item:
                if len(final) == k:
                    break
                final.append(i)

            if len(final) == k:
                break
        x = [0 for i in range(10)]
        for item in final:
            x[item] += 1
        m = max(x)

        if label in [i for i, j in enumerate(x) if j == m]:
            count_correct += 1
            print(count_correct,"/",j)
        print(time.time()-_)






def main():
    # for i in range(1,6):
    #     x = load_training_data(str(i))
    #     print(len(x['data']), type(x['data']))

    x = load_training_data10(1)
    z = []
    z2 = []
    print([key for key in x])

    i = 0
    plot = MultiImage()
    _ = time.time()
    for j in range(len(x['data'])):


        i += 1

        label = x["labels"][j]
        data = x["data"][j]


        data = convert_img(data)

        img = [[i[0]*0.299+i[1]*0.587+i[2]*0.114 for i in j] for j in data]
        plot.add(img, "gray")

        # z.append(crappy_edge_detect(img))
        img = crappy_edge_detect(img)
        plot.add(img, "gray")

        z2.append(label)
        if not i%10:
            plot.draw()
            print(time.time()-_)
            print(i)
            print()
            _ = time.time()
        if not i%1000:
            break
    print("done loading")
    knn(z,z2,7)




if __name__=="__main__":
    main()
