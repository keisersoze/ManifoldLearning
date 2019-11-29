import scipy.io

import scipy.io
from sklearn import manifold

import matplotlib.pyplot as plt

dataset = scipy.io.loadmat('./data/SHOCK.mat')

adjacency_matrices = [cell['am'] for cell in dataset["G"][0] ]
labels = [label[0] for label in dataset["labels"]]

embeddings = manifold.locally_linear_embedding(dataset, n_neighbors=10, n_components=2)

print(dataset)

blockhere = None
