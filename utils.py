import numpy as np
import math


def compute_distance_matrix(kernel_matrix):
    distance_matrix = np.empty(shape=kernel_matrix.shape)
    for (i, j) in np.ndindex(distance_matrix.shape):
        distance_matrix[i, j] = math.sqrt(2 - 2 * kernel_matrix[i, j])
    return distance_matrix

'''
# WIP...
def compute_test_distance_matrix(kernel_matrix, test_matrix):
    res = np.empty(shape=[test_matrix.shape[0],kernel_matrix.shape[0]])
    for (i,j) in np.ndindex(res.shape):
        res[i,j] = math.sqrt(kernel_matrix[i, i] + kernel_matrix[j, j] - 2 * kernel_matrix[i, j])
'''