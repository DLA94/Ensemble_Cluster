#!/usr/local/Cellar python
# _*_coding:utf-8_*_

"""
@Author: 姜小帅
@file: ECS.py
@Time: 2019/11/21 10:18 下午
@Say something:  
# 良好的阶段性收获是坚持的重要动力之一
# 用心做事情，一定会有回报
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans


def calculate_similarity(n, Y):
    """
    Parameters
    ----------
    n : integer
        the number of samples, use to initialize a n*n matrix

    Y : Series, shape (n_samples,)
        the cluster of each sample that model predicted

    Return
    ----------
    similarity_mat : ndarray, shape (n_samples, n_samples)
        transformed by every single result of clustering
    """
    if type(n) != int:

        try:
            n = int(n)
        except TypeError:
            print('Input n should be an integer!')
    elif type(Y) != pd.Series:
        try:
            Y = pd.Series(Y)
        except TypeError:
            print('Input Y should be pd.Series!')

    similarity_mat = np.zeros((n, n))

    clusters = set(Y)
    for cluster in clusters:
        index = Y[Y == cluster].index
        l = np.array(index).reshape(len(index), 1)
        similarity_mat[l, l.T] = 1

    similarity_mat[range(n), range(n)] = 1

    return similarity_mat


def transform(n_round, n_sample, similarity_mat, df):
    """
    Parameters
    ----------
    n_round : integer
        the training times of modeling

    n_sample : integer
        the number of sample, use to initialize a n*n matrix

    Return
    ----------
    similarity_mat : ndarray, shape (n_samples, n_samples)
        a matrix, superimposed every round similarity matrix and standarded
    """
    for i in tqdm(range(n_round)):
        seed = np.random.randint(2019)
        km = KMeans(n_clusters=3, random_state=seed)
        y_pre = km.fit_predict(df[['x', 'y']])
        similarity_mat += calculate_similarity(n_sample, y_pre)
    similarity_mat = similarity_mat / n_round

    return similarity_mat