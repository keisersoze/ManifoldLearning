import scipy.io

import scipy.io
from sklearn import manifold

import matplotlib.pyplot as plt

dataset = scipy.io.loadmat('./data/SHOCK.mat')
cells = dataset["G"][0]

X = []
for cell in cells:
    X.append(cell.base)


embeddings = manifold.locally_linear_embedding(dataset, n_neighbors=10, n_components=2)

print(dataset)

blockhere = None
