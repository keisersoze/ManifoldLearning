import numpy as np
from grakel import GraphKernel
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from datasets_utils import load_shock_dataset
from utils import compute_distance_matrix

X, y = load_shock_dataset()

# Shuffle data
idx = np.random.RandomState(seed=42).permutation(len(X))
X, y = X[idx], y[idx]

# Split indexes according to Kfold with k = 10
k = 10
kf = KFold(n_splits=k)

# initialize scores lists
scores = []

for train_index, test_index in kf.split(X):

    kernel = GraphKernel(kernel={"name": "shortest_path", "with_labels": False}, normalize=True)

    # split train and test of K-fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Calculate the kernel matrix.
    K_train = kernel.fit_transform(X_train)
    K_test = kernel.transform(X_test)

    # Compute distance matrix
    D_train = compute_distance_matrix(K_train)
    D_test = compute_distance_matrix(K_test)

    embedding = manifold.Isomap(n_neighbors=5, n_components=10, metric="precomputed")
    E_train = embedding.fit_transform(D_train)
    E_test = embedding.fit(D_test)  # non esiste ancora
    clf2 = svm.SVC(kernel='linear', C=1)
    clf2.fit(X_train, y_train)
    # Predict and test.
    y_pred = clf2.predict(X_test)
    # Calculate accuracy of classification.
    acc = accuracy_score(y_test, y_pred)
    scores2.append(acc)

    # Initialise an SVM and fit.
    clf = svm.SVC(kernel='precomputed', C=420)
    clf.fit(K_train, y_train)

    # Predict and test.
    y_pred = clf.predict(K_test)

    # Calculate accuracy of classification.
    acc = accuracy_score(y_test, y_pred)

    scores.append(acc)

cross_validation_accuracy = np.mean(scores)

print("Accuracy:", str(round(cross_validation_accuracy*100, 2)), "%")


