import pyspark
import itertools
import sys
import os
import time
import math
import pandas as pd
from surprise import Dataset
from surprise import BaselineOnly
from surprise import Reader

trainfile = sys.argv[1]
testfile = sys.argv[2]
outputfile = sys.argv[3]

if __name__ == '__main__':

    sc_conf = pyspark.SparkConf() \
        .setAppName('inf553') \
        .setMaster('local[*]') \
        .set("spark.driver.memory", "4G") \
        .set("spark.executor.memory", "4G")

    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")
    start = time.time()
    # case = int(sys.argv[3])
    lines = sc.textFile(trainfile)
    header = lines.first()
    lines = lines.filter(lambda x: x != header).map(lambda s: s.split(","))

    users = lines.map(lambda x: (x[0], x[1])).reduceByKey(lambda x, y: 1).map(lambda x: x[0])
    usrs = list(users.collect())
    usrs.sort()
    dic_usr = {}
    for i, e in enumerate(usrs):
        dic_usr[e] = i
    broadcast_usr = sc.broadcast(dic_usr)

    businesses = lines.map(lambda x: (x[1], x[0])).reduceByKey(lambda x, y: 1).map(lambda x: x[0])
    bus = list(businesses.collect())
    bus.sort()
    dic_bus = {}
    for i, e in enumerate(bus):
        dic_bus[e] = i

    matrix = lines.map(lambda x: (x[1], [broadcast_usr.value[x[0]]])).reduceByKey(lambda x, y: x + y).sortBy(
        lambda x: x[0])
    # print(matrix)
    # m: the number of the bins
    m = len(usrs)
    # print(usrs[0])
    # random.seed(123)
    # n is the number of hash functions
    n = 26
    # hash function:
    hashes = [[2, 71], [13, 9, 97], [33, 2], [14, 23, 769], [3, 5, 51], [1, 10, 193], [17, 91, 1543], [43, 1],
              [7, 2], [38, 5, 97], [11, 37, 3079], [81, 4], [2, 63, 97], [42, 24], [41, 67, 1543], [17, 2], [52, 14], \
              [9, 29, 193], [3, 79, 53], [73, 8, 769], [8, 19, 389], [13, 5, 177], [1, 5], [3, 41], [44, 13],
              [22, 3]]


    def f(x, hash):
        a = hash[0]
        b = hash[1]
        # p = 12289
        return min([(a * urs + b) % m for urs in x[1]])
        # return min([((a * e + b) % p) % m for e in x[1]])


    # build signatures
    signatures = matrix.map(lambda x: (x[0], [f(x, hash) for hash in hashes]))
    # b is the number of bands
    b = 13
    r = int(n / b)


    def divide(x):
        # for e in x:
        segmentation = []
        for i in range(b):
            segmentation.append(((i, tuple(x[1][i * r:(i + 1) * r])), [x[0]]))
        return segmentation


    def combine(x):
        can_pairs = []
        candidates = list(x[1])
        candidates.sort()
        pairs = itertools.combinations(candidates, 2)
        for pair in pairs:
            can_pairs.append(((pair[0], pair[1]), 1))
        return can_pairs


    # two items are a candidate pair if their signatures are identical in at least one band.
    candidate = signatures.flatMap(lambda x: divide(x)).reduceByKey(lambda x, y: x + y).filter(
        lambda x: len(x[1]) > 1).flatMap(lambda x: combine(x)) \
        .reduceByKey(lambda x, y: 1).map(lambda x: x[0])
    verify_matrix = matrix.collect()

    user_items = lines.map(lambda x: ((x[0]), ((x[1]), float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    item_users = lines.map(lambda x: ((x[1]), ((x[0]), float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    bro_user_items = sc.broadcast(user_items)
    bro_item_users = sc.broadcast(item_users)
    # CASE4
    lines2 = sc.textFile(testfile)
    header2 = lines2.first()
    test = lines2.filter(lambda x: x != header2).map(lambda s: s.split(",")).map(
        lambda r: ((r[0], r[1]), float(r[2])))
    # CASE1
    train = lines.map(lambda r: ((r[0], r[1]), float(r[2])))
    all_rdd = train.union(test)

    users1 = all_rdd.map(lambda x: (x[0][0], x[0][1])).reduceByKey(lambda x, y: 1).map(lambda x: x[0])
    urs1 = list(users1.collect())
    urs1.sort()
    dict_usr1 = {}
    for i, e in enumerate(urs1):
        dict_usr1[e] = i
    broad_usr1 = sc.broadcast(dict_usr1)

    businesses1 = all_rdd.map(lambda x: (x[0][1], x[0][0])).reduceByKey(lambda x, y: 1).map(lambda x: x[0])
    bus1 = list(businesses1.collect())
    bus1.sort()
    dict_bus1 = {}
    for i, e in enumerate(bus1):
        dict_bus1[e] = i
    broad_bus1 = sc.broadcast(dict_bus1)


    def predict(user1, item1, list_items, list_users):
        user_items = list_items.value
        item_users = list_users.value
        # new user1 and new item1
        if not user_items.get(user1) and not item_users.get(item1):
            return ((user1, item1), 0.0)
        # the user in testing data has past records in training data
        else:
            # user1_items=user_items.get(user1)
            # print(list(user1_items.values))
            # user1_allavg=sum(list(user1_items.values()))/len(list(user1_items))
            # if it is a new item in the training data, so prediction is the avg rating of that user
            if not item_users.get(item1):
                user1_items = user_items.get(user1)
                user1_allavg = sum(list(user1_items.values())) / len(list(user1_items))
                return ((user1, item1), user1_allavg)
            # if it is a new user in the training data, so prediction is the avg rating of that item
            elif not user_items.get(user1):
                item1_users = item_users.get(item1)
                item1_allavg = sum(list(item1_users.values())) / len(list(item1_users))
                return ((user1, item1), item1_allavg)
            # find the users who have already rated the item
            else:
                user1_items = user_items.get(user1)
                item1_users = item_users.get(item1)
                item1_signature = set(verify_matrix[dic_bus[item1]][1])
                item2_weights = []
                # for all other rated item2 by user1, calculate the similarity with item1 using Minhash and LSH
                for item2 in list(user1_items):
                    item2_signature = set(verify_matrix[dic_bus[item2]][1])
                    inter = item1_signature & item2_signature
                    union = item1_signature | item2_signature
                    jacc = len(inter) / len(union)
                    if jacc >0.9:
                        item2_users = item_users.get(item2)
                        ratings1, ratings2 = [], []
                        # find user who has rated both item1 and item2
                        for item1_user in item1_users:
                            if item1_user in item2_users:
                                ratings1.append(item1_users.get(item1_user))
                                ratings2.append(item2_users.get(item1_user))
                        # if len(ratings2)==0:
                        #     item2_weights.append((0, 0))
                        if len(ratings2) != 0:
                            # item1_avg = sum(list(item1_ratings.values())) / len(list(item1_ratings))
                            # item2_avg = sum(list(item2_ratings.values())) / len(list(item2_ratings))
                            item1_avg = sum(ratings1) / len(ratings1)
                            item2_avg = sum(ratings2) / len(ratings2)
                            up, down = 0, 0
                            item1_sum, item2_sum = 0, 0
                            for i in range(len(ratings1)):
                                up += (ratings1[i] - item1_avg) * (ratings2[i] - item2_avg)
                                item1_sum += (ratings1[i] - item1_avg) ** 2
                                item2_sum += (ratings2[i] - item2_avg) ** 2
                            down = math.sqrt(item1_sum) * math.sqrt(item2_sum)
                            if down != 0:
                                weight1 = (up / down)*jacc
                                weight = weight1 * (abs(weight1) ** (1.5))
                                # user2_item_rating=user_items[user2].get(item1)
                                #
                                item2_weights.append((item2_users.get(user1) * weight, weight))
                            # else:
                            #     item2_weights.append((0, 0))
                        # print(len(list(user1_items)),'&&',len(item2_weights))
                    up2, down2 = 0, 0
                    # the neighborhood N=5
                    cc=1
                    item2_weights.sort(key=lambda y: y[1], reverse=True)
                    # print(item2_weights)
                    for item2_weight in item2_weights[:]:
                        if item2_weight[1] > 0 and cc<=10:
                            up2 += item2_weight[0]
                            down2 += abs(item2_weight[1])
                            cc+=1
                        else:
                            break
                    if down2 != 0:
                        prediction = up2 / down2
                        res = prediction
                        if prediction >= 5:
                            res = 5
                        elif prediction <= 1:
                            res = 1
                        return ((user1, item1), res)
                    else:
                        # item1_users = item_users.get(item1)
                        # item1_allavg = sum(list(item1_users.values())) / len(list(item1_users))
                        item1_allavg = sum(list(user1_items.values())) / len(list(user1_items))
                        return ((user1, item1), item1_allavg)


    test_key = test.map(lambda x: (x[0][0], x[0][1]))
    predictions2 = test_key.map(lambda x: predict(x[0], x[1], bro_user_items, bro_item_users)).persist()

    # Bulid ALS model by suprise
    df1 = pd.read_csv(trainfile)
    reader = Reader()
    traindata = Dataset.load_from_df(df1, reader)
    trainset = traindata.build_full_trainset()

    df2 = pd.read_csv(testfile)
    reader2 = Reader()
    testdata = Dataset.load_from_df(df2, reader2)
    testset = testdata.construct_testset(testdata.raw_ratings)

    bsl_options = {'method': 'als',
                   'n_epochs': 200,
                   'learning_rate': .005,
                   'reg_u': 0.2,
                   'reg_i': 0.1,
                   'user_based': False,
                   'name': 'pearson_baseline',
                   'min_support': 2,
                   'shrinkage': 200
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    algo.fit(trainset)
    predictions1 = algo.test(testset)
    # accuracy.rmse(predictions, verbose=True)
    # pred_rdd = sc.parallelize(predictions1).map(lambda x: ((x[0], x[1]), x[3]))
    pred_rdd = sc.parallelize(predictions1).map(lambda x: ((broad_usr1.value[x[0]], broad_bus1.value[x[1]]), x[3]))
    pred= predictions2.map(lambda x: ((broad_usr1.value[x[0][0]], broad_bus1.value[x[0][1]]), x[1])). \
        leftOuterJoin(pred_rdd).mapValues(lambda x: x[0] if x[1] == None else 0.95*x[1]+0.05*x[0])
    
    trueAndPreds = test.map(lambda x: ((broad_usr1.value[x[0][0]], broad_bus1.value[x[0][1]]), x[1])).leftOuterJoin(pred)
    # trueAndPreds = test.join(preds3)
    MSE = trueAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()

    print("Mean Squared Error = " + ('%.5f' % math.sqrt(MSE)))

    diff = trueAndPreds.map(lambda r: abs(r[1][0] - r[1][1]))
    diff01 = diff.filter(lambda x: 0 <= x < 1)
    diff12 = diff.filter(lambda x: 1 <= x < 2)
    diff23 = diff.filter(lambda x: 2 <= x < 3)
    diff34 = diff.filter(lambda x: 3 <= x < 4)
    diff4 = diff.filter(lambda x: 4 <= x)
    print(">=0 and <1: " + str(diff01.count()))
    print(">=1 and <2: " + str(diff12.count()))
    print(">=2 and <3: " + str(diff23.count()))
    print(">=3 and <4: " + str(diff34.count()))
    print(">=4: " + str(diff4.count()))

    path = os.path.join(outputfile)
    with open(path, "w") as testFile:
        testFile.write("user_id, business_id, prediction\n")
        for p in trueAndPreds.collect():
            testFile.write(urs1[p[0][0]] + "," + bus1[p[0][1]] + "," + str(p[1][1]) + "\n")

    end = time.time()
    print(end - start)
