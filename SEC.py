#!/usr/local/Cellar python
# -*- coding: utf-8 -*-
"""
@File Name: ensemble.py
@Author: 姜小帅
@Motto: 良好的阶段性收获是坚持的重要动力之一
@Date: 2020/2/19
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ensemble_Cluster.vat import VAT
from sklearn.cluster import KMeans, SpectralClustering


class SEC:
    def __init__(self, df, n_round, remove=0):
        self.df = df
        self.remove = remove
        self.n_round = n_round
        self.n_sample = df.shape[0]
        self.similarity_mat = np.zeros((self.n_sample, self.n_sample))

    def calculate(self):

        for i in range(self.n_round):
            # one round for one similarity matrix
            # initial a similarity matrix
            mat = np.zeros((self.n_sample, self.n_sample))
            # number of clusters is a random int between 5 and Radical n
            n = random.randint(5, int(np.sqrt(self.n_sample)))
            km = KMeans(n_clusters=n)
            self.df['pre'] = km.fit_predict(self.df[['x', 'y']])
            Y = self.df['pre']
            clusters = set(Y)
            # update similarity matrix
            for cluster in clusters:
                index = Y[Y == cluster].index
                l = np.array(index).reshape(1, -1)
                mat[l, l.T] = 1 / self.n_round

            self.similarity_mat += np.array(mat)

        return self.df, self.similarity_mat


if __name__ == '__main__':

    path = '.../data/'
    df = pd.read_csv(path + '_3Gaussians.csv', names=['x', 'y'])

    en = SEC(n_round=1000)
    df, mat = en.calculate()
    # spectral clustering, parameter affinity='precomputed' means
    # that you could dentify your affinity matrix
    sc = SpectralClustering(n_clusters=4, affinity='precomputed', assign_labels='kmeans')
    df['sc_pre'] = sc.fit_predict(mat)

    colors = ['r', 'b', 'black', 'pink', 'g', 'grey', 'purple']
    plt.figure(figsize=(6, 6))
    for i in list(set(df['sc_pre'])):
        temp = df[df['sc_pre'] == i]
        plt.scatter(temp['x'], temp['y'], c=colors[i])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    vat = VAT(mat)
    vat.plot()