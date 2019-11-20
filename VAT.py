#!/usr/local/Cellar python
# _*_coding:utf-8_*_

"""
@Author: 姜小帅
@file: VAT.py
@Time: 2019/11/19 11:06 下午
@Say something:  
# 良好的阶段性收获是坚持的重要动力之一
# 用心做事情，一定会有回报
"""
import numpy as np
import matplotlib.pyplot as plt

def VAT(similarity_matrix):
    P = []
    I = set()
    K = set(range(len(similarity_matrix)))
    J = K

    max_element_row = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)[0]
    P.append(max_element_row)
    I.add(max_element_row)
    J = J.difference(I)

    while len(J)!=0:
        
        m = np.array(list(I)).reshape(len(I), 1)
        n = np.array(list(J)).reshape(len(J), 1)
        temp = similarity_matrix[m, n.T]
        
        min_element_col = np.unravel_index(temp.argmin(), temp.shape)[1]
        I.add(n[min_element_col][0])
        P.append(n[min_element_col][0])
        J = J.difference(I)

    ordered_array = similarity_matrix[:,P][P,:]
    plt.imshow(ordered_array, cmap='inferno')
    plt.show()

    return ordered_array
    

if __name__ == "__main__":
    similarity_mat = np.array([[0, 0.73, 0.19, 0.71, 0.16], [0.73, 0, 0.59, 0.12, 0.78], [0.19, 0.59, 0, 0.55, 0.19], \
              [0.71, 0.12, 0.55, 0, 0.74], [0.16, 0.78, 0.19, 0.74, 0]])

    ordered_array = VAT(similarity_mat)
    print('ordered_array equals:')
    print(ordered_array)
