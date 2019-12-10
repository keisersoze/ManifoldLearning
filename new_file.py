import matplotlib
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

from datasets_utils import load_shock_dataset

X, y = load_shock_dataset()

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

embedding = manifold.Isomap(n_neighbors=5, n_components=2, metric="precomputed")
X_transformed = embedding.fit_transform(D)

# xs = feature_vectors[:, 0]
# ys = feature_vectors[:, 1]
#
# plt.scatter(xs, ys, c=y)
# plt.show()

# print(np.all(np.isfinite(K_train)))

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)

# Initialise an SVM and fit.
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)


# Predict and test.
y_pred = clf.predict(X_test)

# Calculate accuracy of classification.
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", str(round(acc * 100, 2)))

# embeddings = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=2)

# blockhere = None
