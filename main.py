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

# Calculate the kernel matrix for random Walk Kernel.
K_train = randomWalkKernel.fit_transform(X_train)
K_test = randomWalkKernel.transform(X_test)
'''nanel = 0

print (K_train[0][79-5])
print(len(K_train))
print(len(K_train[0]))
for i in K_train:
    for el in i:
        if np.isnan(el):
            nanel += 1
print("\n How many nan elements are there? Are there exactly len(K_train) elements? ",nanel, nanel == len(K_train))
'''
# There are 158 nan elements in K_train
# https://github.com/ysig/GraKeL/issues/6
# I transform each nan element into a number

K_train = np.nan_to_num(K_train)
K_test = np.nan_to_num(K_test)

#nan_elements = np.any(np.isnan(K_train))

#print(np.all(np.isfinite(K_train)))

# Initialise an SVM and fit data using random walk Kernel.
clf = svm.SVC(kernel='precomputed', C=1)
clf.fit(K_train, y_train)

# Predict and test.
y_pred = clf.predict(K_test)

# Calculate accuracy of classification.
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", str(round(acc * 100, 2)))

K_train = manifold.locally_linear_embedding(K_train, n_neighbors=5, n_components=3)
K_test = manifold.locally_linear_embedding(K_test, n_neighbors=5, n_components=3)

clf.fit(K_train, y_train, )

# Predict and test.
y_pred = clf.predict(K_test)

# Calculate accuracy of classification.
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", str(round(acc * 100, 2)))

blockhere = None

dataset = scipy.io.loadmat('./data/PPI.mat')

adjacency_matrices = np.array([[cell['am']] for cell in dataset["G"][0]])
labels = np.array([label[0] for label in dataset["labels"]])

X = adjacency_matrices
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=train_size,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    random_state=42)
