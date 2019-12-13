import scipy.io
import numpy as np


def load_shock_dataset():
    dataset_shock = scipy.io.loadmat('./data/SHOCK.mat')
    adjacency_matrices_shock = np.array([[cell['am']] for cell in dataset_shock["G"][0]])
    labels_shock = np.array([label[0] for label in dataset_shock["labels"]])
    return adjacency_matrices_shock, labels_shock


def load_ppi_dataset():
    dataset_ppi = scipy.io.loadmat('./data/PPI.mat')
    adjacency_matrices_ppi = np.array([[cell['am']] for cell in dataset_ppi["G"][0]])
    labels_ppi = np.array([label[0] for label in dataset_ppi["labels"]])
    return adjacency_matrices_ppi, labels_ppi