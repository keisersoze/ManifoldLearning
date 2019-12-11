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
from utils import compute_distance_matrix

X, y = load_shock_dataset()

## KFOLD
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

spk = GraphKernel(kernel={"name": "shortest_path", "with_labels": False}, normalize=True)

from sklearn.model_selection import KFold
kf = KFold(n_splits=2) # n_splits is 2 for testing reasons

scores = []
for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    K_train = spk.fit_transform(X_train)
    K_test = spk.transform(X_test)

    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)

mean = 0
print(scores)
for x in scores:
    mean += x
mean /= len(scores)
print("Accuracy: %0.2f" % (mean) )#, scores.std() * 2))

##END KFOLD

shortestPathKernel = GraphKernel(kernel={"name": "shortest_path", "with_labels": False}, normalize=True)


# Calculate the kernel matrix.
K = shortestPathKernel.fit_transform(X)

nan_elements = np.any(np.isnan(K))

# Compute the distance matrix D
D = compute_distance_matrix(K)

embedding = manifold.Isomap(n_neighbors=5, n_components=10, metric="precomputed")
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
