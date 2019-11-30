import scipy.io

from sklearn.model_selection import train_test_split
from sklearn import manifold
from sklearn import svm
from sklearn.metrics import accuracy_score

from grakel import GraphKernel

import numpy as np

import matplotlib.pyplot as plt

dataset = scipy.io.loadmat('./data/SHOCK.mat')

train_size = 80
test_size = 10
adjacency_matrices = np.array([[cell['am']] for cell in dataset["G"][0]])
labels = np.array([label[0] for label in dataset["labels"]])

X = adjacency_matrices
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=train_size,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    random_state=42)

randomWalkKernel = GraphKernel(kernel={"name": "random_walk", "with_labels": False}, normalize=True)
graphletKernel = GraphKernel(kernel={"name": "graphlet_sampling"}, normalize=True)
shortestPathKernel = GraphKernel(kernel={"name": "shortest_path"}, normalize=True)

# Calculate the kernel matrix.
K_train = randomWalkKernel.fit_transform(X_train)
K_test = randomWalkKernel.transform(X_test)

nan_elements = np.any(np.isnan(K_train))
# print(np.all(np.isfinite(K_train)))

# Initialise an SVM and fit.
clf = svm.SVC(kernel='precomputed', C=1)
clf.fit(K_train, y_train)

# Predict and test.
y_pred = clf.predict(K_test)

# Calculate accuracy of classification.
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", str(round(acc * 100, 2)))

# embeddings = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=2)

blockhere = None
