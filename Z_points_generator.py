import numpy as np
from grakel import GraphKernel
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import manifold
from datasets_utils import load_shock_dataset
from datasets_utils import load_ppi_dataset
from utils import compute_distance_matrix
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D



X, y = load_shock_dataset()
#X, y = load_ppi_dataset()

# Shuffle data
idx = np.random.RandomState(seed=42).permutation(len(X))
X, y = X[idx], y[idx]

# initialize scores lists

def spk_isomap(X,y, k, KNNstart, KNNend, Dstart, Dend, svmC):

    filename = "accuracy.txt"

    myfile = open(filename, 'a')

    # Add info to file
    myfile.write('SP Isomap accuracy: K = %d-%d, D = %d-%d, C = %d, K-fold = %d\n'
                 % (KNNstart, KNNend, Dstart, Dend, svmC, k))

    KNN = []
    KNNrange = KNNend - KNNstart+1
    D = []
    Drange = Dend - Dstart+1

    for knn in range(KNNrange):
        KNN.append( knn + KNNstart)


    for d in range(Drange):
        D.append(d + Dstart)


    kf = KFold(n_splits=k)
    scores = []

    Z = np.ndarray(shape=( len(D) , len(KNN) ))

    for knn in range(len(KNN)):
        for d in range(len(D)):

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

                # Initialize Isomap embedding object, embed train and test data
                embedding = manifold.Isomap(n_neighbors=KNN[knn], n_components=D[d], metric="precomputed")
                E_train = embedding.fit_transform(D_train)
                E_test = embedding.transform(D_test)

                # initialize second svm (not necessary? search documentation)
                clf2 = svm.SVC(kernel='linear', C=svmC)
                clf2.fit(E_train, y_train)

                # Predict and test.
                y_pred = clf2.predict(E_test)

                # Append accuracy of classification.
                scores.append(accuracy_score(y_test, y_pred))

            val = np.mean(scores)
            Z[d][knn] = val
            myfile.write("%f " % (val))
            print("knn = ", KNN[knn], "d = ", D[d], " accuracy = ", Z[d][knn])
            print("{0:.2%} done".format((Drange*knn+d+1.0)/(Drange*KNNrange)))
            # print("{0:.2%} done".format((D*k+d + 1.0)/(D*KNN) ))
        myfile.write("\n")
    # Close the file
    myfile.close()
    return Z
