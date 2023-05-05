import operator

import pandas as pd
import numpy as np
from mygethash import gethash

Max_inf = 0x3f3f3f3f
N = 500  # hash函数的个数
K = 30  # 选取相似用户的个数
top_n = 10
The_user = 17
Primer = 1009
rows = 180000
cols = 1000

Train_csv = './data/train_set.csv'
Test_csv = './data/test_set.csv'

Train_pd = pd.read_csv(Train_csv)
Test_pd = pd.read_csv(Test_csv)

Utility = np.zeros((rows, cols))
Feature = np.zeros((rows, cols))

re_rows = 0
re_cols = 0
for k in range(Train_pd.shape[0]):
    i = Train_pd['userId'][k]
    re_cols = max(i, re_cols)
    j = Train_pd['movieId'][k]
    re_rows = max(j, re_rows)
    rate = Train_pd['rating'][k]
    Utility[j - 1, i - 1] = rate
    Feature[j - 1, i - 1] = 0 if rate < 3.0 else 1

Feature = Feature[0:re_rows, 0:re_cols]
rows = re_rows
cols = re_cols
# print(Utility)
print(Utility[:, 0])
print(Utility[:, 1])
print(Feature.shape)

Hashlist = [gethash(Primer, N) for i in range(0, N)]
SigMatrix = np.full((N, cols), Max_inf)

for i in range(0, Feature.shape[0]):
    for j in range(0, Feature.shape[1]):
        if Feature[i, j] == 1:
            for k in range(0, N):
                hash_pos = Hashlist[k](i)
                if hash_pos < SigMatrix[k, j]:
                    SigMatrix[k, j] = hash_pos
print(SigMatrix[:, 0])
# np.save('./tmp/save_SigMatrix', SigMatrix)
# SigMatrix = np.load('./tmp/save_SigMatrix.npy')
print(SigMatrix[:, 1])
print(SigMatrix.shape)


def recommened_user(user_id):
    user_id = user_id - 1
    sim_dict = {}
    for col in range(0, re_cols):
        if user_id == col:
            continue
        n = 0
        for row in range(0, N):
            if SigMatrix[row, col] == SigMatrix[row, user_id]:
                n += 1
        sim = n / N
        sim_dict[col + 1] = sim
    sim_list = sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)
    top_k = sim_list[0:K]
    print(top_k)
    all_moive = {}
    for i in range(0, re_cols):
        sum_sim = 0
        sum_score = 0
        if Utility[i, user_id] > 0:
            continue
        for j, sim in top_k:
            if Utility[i, j - 1] == 0:
                continue
            sum_sim += sim
            sum_score += (sim * Utility[i, j - 1])
        if sum_sim == 0:
            all_moive[i + 1] = 0
        else:
            all_moive[i + 1] = sum_score / sum_sim
    new_all = sorted(all_moive.items(), key=operator.itemgetter(1), reverse=True)
    return new_all[0:top_n]


def recommend_rate():
    SSE = 0
    user_set = set()
    for i in range(0, Test_pd.shape[0]):
        user_set.add(Test_pd['userId'][i])
    for user in user_set:

        user = user - 1
        sim_dict = {}
        for col in range(0, re_cols):
            if user == col:
                continue
            n = 0
            for row in range(0, N):
                if SigMatrix[row, col] == SigMatrix[row, user]:
                    n += 1
            sim = n / N
            sim_dict[col + 1] = sim
        sim_list = sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)
        top_k = sim_list[0:K]
        # print(top_k)

        user_to_movies = Test_pd[Test_pd['userId'] == user + 1]['movieId']
        user_to_rates = Test_pd[Test_pd['userId'] == user + 1]['rating']
        for i in range(0, user_to_movies.shape[0]):
            sum_sim = 0
            sum_score = 0
            for j, sim in top_k:
                if Utility[int(user_to_movies.iloc[i]), j - 1] == 0:
                    continue
                sum_sim += sim
                sum_score += (sim * Utility[int(user_to_movies.iloc[i]), j - 1])
            if sum_sim == 0:
                a = 0
            else:
                a = sum_score / sum_sim
            SSE += (user_to_rates.iloc[i] - a) ** 2
            print(user_to_rates.iloc[i], a)
    return SSE


print(recommened_user(4))
print(recommend_rate())
