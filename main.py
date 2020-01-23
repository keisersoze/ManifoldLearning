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

#X, y = load_shock_dataset()
X, y = load_ppi_dataset()

# Shuffle data
idx = np.random.RandomState(seed=42).permutation(len(X))
X, y = X[idx], y[idx]

# Split indexes according to Kfold with k = 10
k = 5
kf = KFold(n_splits=k)
# Hyperparameters
knn = 20
d = 30
Csvm = 100

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
    clf = svm.SVC(kernel='precomputed', C=Csvm)
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
    embedding = manifold.Isomap(n_neighbors=knn, n_components=d, metric="precomputed")
    E_train = embedding.fit_transform(D_train)
    E_test = embedding.transform(D_test)

    # initialize second svm (not necessary? search documentation)
    clf2 = svm.SVC(kernel='linear', C=Csvm)
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

no_manifold_accuracy = np.mean(scores)
with_manifold_accuracy = np.mean(scores2)
no_manifold_se = stats.sem(scores)
with_manifold_se = stats.sem(scores2)

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