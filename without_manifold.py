from grakel import GraphKernel
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from datasets_utils import load_shock_dataset, load_ppi_dataset
from matplotlib import pylab as pl


X, y = load_shock_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=48)

shortestPathKernel = GraphKernel(kernel={"name": "shortest_path", "with_labels": False}, normalize=True)
randomWalkKernel = GraphKernel(kernel={"name": "random_walk", "with_labels": False}, normalize=True)
graphletKernel = GraphKernel(kernel={"name": "graphlet_sampling"}, normalize=True)

kernel = shortestPathKernel

# Calculate the kernel matrix.
K_train = kernel.fit_transform(X_train)
K_test = kernel.transform(X_test)

fig = pl.figure()
pl.subplot(121)
pl.imshow(K_train)
pl.subplot(122)
pl.imshow(K_test)
pl.show()

# Initialise an SVM and fit.
clf = svm.SVC(kernel='precomputed', C=1)
clf.fit(K_train, y_train)

# Predict and test.
y_pred = clf.predict(K_test)

# Calculate accuracy of classification.
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", str(round(acc*100, 2)), "%")


