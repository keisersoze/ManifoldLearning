import matplotlib
import scipy.io

from sklearn.model_selection import train_test_split
from sklearn import manifold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import math

from grakel import GraphKernel

import numpy as np

import matplotlib.pyplot as plt

from datasets_utils import load_shock_dataset,load_ppi_dataset
from utils import compute_distance_matrix


X, y = load_shock_dataset()
# X, y = load_ppi_dataset()

## KFOLD

# Shuffle data
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Initialize chosen Kernel
spk = GraphKernel(kernel={"name": "shortest_path", "with_labels": False}, normalize=True)

# Split indexes according to Kfold with k = 10
k = 10
kf = KFold(n_splits=k)

# initialize scores lists
scores1 = []
scores2 = []


for train_index, test_index in kf.split(X):

    # split train and test of K-fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit and transform train and test with the graph kernel
    K_train = spk.fit_transform(X_train)
    K_test = spk.transform(X_test)

    # Initialize and fit classifier for non-embedded graph with test data
    clf1 = svm.SVC(kernel='linear', C=1)
    clf1.fit(K_train, y_train)

    # make prediction and calculate accuracy
    y_pred = clf1.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    scores1.append(acc)

    '''
    D_train = compute_distance_matrix(K_train)
    D_test = compute_test_distance_matrix (K_train, K_test)
    embedding = manifold.Isomap(n_neighbors=5, n_components=10, metric="precomputed")
    E_train = embedding.fit_transform(D_train)
    E_test = embedding.fit(D_test) # non esiste ancora
    clf2 = svm.SVC(kernel='linear', C=1)
    clf2.fit(X_train, y_train)
    # Predict and test.
    y_pred = clf2.predict(X_test)
    # Calculate accuracy of classification.
    acc = accuracy_score(y_test, y_pred)
    scores2.append(acc)
    '''
# Calculate mean of the scores
mean1 = 0
for x in scores1:
    mean1 += x
mean1 /= k

print("Accuracy of K-Fold non-embedded classification: %0.3f" % (mean1) ) #, scores.std() * 2)) # should calculate std for scores
print("Accuracy of K-Fold embedded classification: NOT YET AVALIABLE") #%0.3f" % (mean2) )

##END KFOLD

##OLD CODE

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

print("Accuracy of embedded classification:", str(round(acc * 100, 2)))

# embeddings = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=2)

# blockhere = None
