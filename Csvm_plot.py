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
from matplotlib import pyplot

X, y = load_shock_dataset()
#X, y = load_ppi_dataset()

# Shuffle data
idx = np.random.RandomState(seed=42).permutation(len(X))
X, y = X[idx], y[idx]

# Split indexes according to Kfold with k = 10

# Hyperparameters
Csvm_start = 10
Csvm_end= 20
k = 10
def Csvm_SPK(X,y,Csvm_start, Csvm_end, k, fun = lambda x : x):
    Csvm_range = Csvm_end-Csvm_start+1
    res = []
    x_points = []
    for c in range(Csvm_range):
        Csvm = fun(c+Csvm_start)
        # initialize scores list
        scores = []
        # initialize x-axis points


        kf = KFold(n_splits=k)
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

        res.append( np.mean(scores))
        x_points.append(fun(c + Csvm_start))
        print("{0:.2%} done".format((c+1.0)/Csvm_range))

    pyplot.plot(x_points, res, 'ro')
    pyplot.title("%d - fold avg. accuracy of SVM over C without ML step" %(k))
    pyplot.xlabel('C')
    pyplot.ylabel('Avg. accuracy')
    pyplot.show()

Csvm_SPK(X,y,Csvm_start,Csvm_end, k, lambda x: 10*(x) )


