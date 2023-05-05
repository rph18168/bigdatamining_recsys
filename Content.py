import numpy as np
import pandas
import pandas as pd
from mygethash import gethash

movies_csv = './data/movies.csv'
train_csv = './data/train_set.csv'
test_csv = './data/test_set.csv'

N = 8
Primer = 13
Max_inf = 0x3f3f3f3f


# movies_pd = pandas.read_csv(movies_csv)

def get_tfidf_feturematrix(movies_path):
    pd = pandas.read_csv(movies_path)
    tf_dict = {}
    idf_dict = {}
    re_rows = 0
    for i in range(0, pd.shape[0]):
        m_id = pd['movieId'][i]
        re_rows = max(re_rows, m_id)
        f_string = pd['genres'][i]
        f_list = f_string.split('|')
        f_len = len(f_list)
        tf_dict[m_id] = {}
        for feture in f_list:
            tf_dict[m_id][feture] = 1 / f_len
            idf_dict.setdefault(feture, 0)
            idf_dict[feture] += 1
    feture_order_dict = {}
    i = 0
    for key in idf_dict.keys():
        feture_order_dict[key] = i
        i += 1


def get_feturematrix(movies_path):
    pd = pandas.read_csv(movies_path)
    feture_set = set()
    feture_dict = {}
    re_cols = 0
    for i in range(0, pd.shape[0]):
        m_id = pd['movieId'][i]
        re_cols = max(re_cols, m_id)
        f_string = pd['genres'][i]
        f_list = f_string.split('|')
        feture_dict[m_id] = f_list
        for feture in f_list:
            feture_set.add(feture)
    f_len = len(feture_set)
    i = 0
    order_dict = {}
    for feture in feture_set:
        order_dict[feture] = i
        i += 1
    feturematrix = np.zeros((f_len, re_cols), dtype=int)
    for key_id in feture_dict.keys():
        for feture in feture_dict[key_id]:
            feturematrix[order_dict[feture], key_id - 1] = 1
    return feturematrix


def minihash(f_matrix):
    Hashlist = [gethash(Primer, N) for i in range(0, N)]
    SigMatrix = np.full((N, f_matrix.shape[1]), Max_inf)

    for i in range(0, f_matrix.shape[0]):
        for j in range(0, f_matrix.shape[1]):
            if f_matrix[i, j] == 1:
                for k in range(0, N):
                    hash_pos = Hashlist[k](i)
                    if hash_pos < SigMatrix[k, j]:
                        SigMatrix[k, j] = hash_pos
    return SigMatrix


def pre_score(Sigmatrix, user, movieId, rate, df):
    movie_Ids = df[df['userId'] == user]['movieId']
    movie_rateings = df[df['userId'] == user]['rating']
    sum_score = 0
    sum_sim = 0
    for i in range(0, movie_Ids.shape[0]):
        m_id = movie_Ids.iloc[i]
        n = 0
        for j in range(0, N):
            if Sigmatrix[j, m_id - 1] == Sigmatrix[j, movieId - 1]:
                n += 1
        sim = n / N
        sum_sim += sim
        sum_score += (sim * movie_rateings.iloc[i])
    pre_rate = sum_score / sum_sim
    return rate, pre_rate, (pre_rate - rate) ** 2


f_matrix = get_feturematrix(movies_path=movies_csv)
# print(f_matrix)
Sigmatrix = minihash(f_matrix)
# print(Sigmatrix)
# print(f_matrix[:, 0])
# print(f_matrix[:, 1])
# print(Sigmatrix[:, 0])
# print(Sigmatrix[:, 1])
df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
for i in range(0, test_df.shape[0]):
    print(
        pre_score(Sigmatrix, test_df['userId'][i], test_df['movieId'][i], test_df['rating'][i], pd.read_csv(train_csv)))
