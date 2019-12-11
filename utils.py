import numpy as np
import math


def compute_distance_matrix(kernel_matrix):
    distance_matrix = np.empty(shape=kernel_matrix.shape)
    for (i, j) in np.ndindex(distance_matrix.shape):
        distance_matrix[i, j] = math.sqrt(kernel_matrix[i, i] + kernel_matrix[j, j] - 2 * kernel_matrix[i, j])
    return distance_matrix
