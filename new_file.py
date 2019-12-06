import scipy.io

from sklearn.model_selection import train_test_split
from sklearn import manifold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

import math

from grakel import GraphKernel

import numpy as np


import matplotlib.pyplot as plt

dataset_shock = scipy.io.loadmat('./data/SHOCK.mat')
dataset_ppi = scipy.io.loadmat('./data/PPI.mat')

adjacency_matrices_shock = np.array([[cell['am']] for cell in dataset_shock["G"][0]])
labels_shock = np.array([label[0] for label in dataset_shock["labels"]])

adjacency_matrices_ppi = np.array([[cell['am']] for cell in dataset_ppi["G"][0]])
labels_ppi = np.array([label[0] for label in dataset_ppi["labels"]])

X = adjacency_matrices_shock
y = labels_shock

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.2,
#                                                     shuffle=True,
#                                                     random_state=42)

randomWalkKernel = GraphKernel(kernel={"name": "random_walk", "with_labels": False}, normalize=True)
graphletKernel = GraphKernel(kernel={"name": "graphlet_sampling", "with_labels": False}, normalize=True)
shortestPathKernel = GraphKernel(kernel={"name": "shortest_path", "with_labels": False}, normalize=True)

# Calculate the kernel matrix.
K = shortestPathKernel.fit_transform(X)

nan_elements = np.any(np.isnan(K))

# Compute the distance matrix D
D = np.empty(shape=K.shape)

for (i, j) in np.ndindex(D.shape):
    D[i, j] = math.sqrt(K[i, i] + K[j, j] - 2 * K[i, j])

neigh = NearestNeighbors(n_neighbors = 5, metric="precomputed")
neigh.fit(D)


# print(np.all(np.isfinite(K_train)))

# Initialise an SVM and fit.
clf = svm.SVC(kernel='precomputed', C=1)
clf.fit(K, y)

# Predict and test.
y_pred = clf.predict(K_test)

# Calculate accuracy of classification.
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", str(round(acc * 100, 2)))

# embeddings = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=2)

blockhere = None
