import numpy as np
import scipy
from grakel import GraphKernel
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import manifold
from datasets_utils import load_shock_dataset, load_ppi_dataset
from utils import compute_distance_matrix
from scipy import stats
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from random import random


def cross_validation_with_and_without_manifold(X, y, n_neighbors, n_components, k):
    # Split indexes according to Kfold with k = 10
    kf = KFold(n_splits=k)

    # initialize scores lists
    scores = []
    scores2 = []
    for train_index, test_index in kf.split(X):
        kernel = GraphKernel(kernel={"name": "shortest_path", "with_labels": False}, normalize=True)

        # split train and test of K-fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Calculate the kernel matrix.
        K_train = kernel.fit_transform(X_train)
        K_test = kernel.transform(X_test)

        # Initialise an SVM and fit.
        clf = svm.SVC(kernel='precomputed', C=4)
        clf.fit(K_train, y_train)

        # Predict and test.
        y_pred = clf.predict(K_test)

        # Calculate accuracy of classification.
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

        # Compute distance matrix
        D_train = compute_distance_matrix(K_train)
        D_test = compute_distance_matrix(K_test)

        # Initialize Isomap embedding object, embed train and test data
        embedding = manifold.Isomap(n_neighbors, n_components, metric="precomputed")
        E_train = embedding.fit_transform(D_train)
        E_test = embedding.transform(D_test)

        # initialize second svm (not necessary? search documentation)
        clf2 = svm.SVC(kernel='linear', C=4)
        clf2.fit(E_train, y_train)

        # Predict and test.
        y_pred = clf2.predict(E_test)

        # Calculate accuracy of classification.
        acc = accuracy_score(y_test, y_pred)
        scores2.append(acc)
    for i, _ in enumerate(scores):
        scores[i] = scores[i] * 100

    for i, _ in enumerate(scores2):
        scores2[i] = scores2[i] * 100
    return scores, scores2


X, y = load_shock_dataset()

# Shuffle data
idx = np.random.RandomState(seed=42).permutation(len(X))
X, y = X[idx], y[idx]

min_dim = 5
max_dim = 35

min_neighbors = 5
max_neighbors = 20

n_components_array = np.arange(min_dim, max_dim + 1, 1)
n_neighbors_array = np.arange(min_neighbors, max_neighbors + 1, 1)
accuracy_matrix = np.zeros((len(n_neighbors_array), len(n_components_array)))

for i, n_neighbors in enumerate(n_neighbors_array):
    for j, n_components in enumerate(n_components_array):
        scores, scores2 = cross_validation_with_and_without_manifold(X, y, n_neighbors=n_neighbors,
                                                                     n_components=n_components, k=2)
        no_manifold_accuracy = np.mean(scores)
        with_manifold_accuracy = np.mean(scores2)
        no_manifold_se = stats.sem(scores)
        with_manifold_se = stats.sem(scores2)
        accuracy_matrix[i, j] = with_manifold_accuracy

X2, Y2 = np.meshgrid(n_components_array, n_neighbors_array)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

mycmap = cm.coolwarm

surf = ax.plot_surface(X2, Y2, accuracy_matrix, rstride=1, cstride=1, alpha=0.6, cmap=mycmap)
cset = ax.contourf(X2, Y2, accuracy_matrix, zdir='z', offset=np.min(accuracy_matrix) - 31, cmap=mycmap)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('dimensions')
ax.set_xlim(min_dim, max_dim )
ax.set_ylabel('#neighbors')
ax.set_ylim(min_neighbors, max_neighbors)
ax.set_zlabel('accuracy %')
ax.set_zlim(np.min(accuracy_matrix) - 31, np.max(accuracy_matrix))
ax.set_title('3D surface with 2D contour plot projections')

plt.show()

print("Accuracy of K-Fold non-embedded classification: %0.3f, SE= +/- %0.3f" % (no_manifold_accuracy, no_manifold_se) )# , scores.std() * 2)) # should calculate std for scores
print("Accuracy of K-Fold embedded classification: %0.3f , SE= +/- %0.3f" % (with_manifold_accuracy, with_manifold_se) )
print("knn = ", knn, "d = ",d, "Csvm = ", Csvm)
#print("Accuracy:", str(round(cross_validation_accuracy*100, 2)), "%")

# PPI Results:
# Accuracy of K-Fold non-embedded classification: 0.674, SE= +/- 0.050
# Accuracy of K-Fold embedded classification: 0.660 , SE= +/- 0.049

# SHOCK Results:
#Accuracy of K-Fold non-embedded classification: 0.407, SE= +/- 0.034
#Accuracy of K-Fold embedded classification: 0.433 , SE= +/- 0.023
#knn =  25 d =  40 Csvm =  100
