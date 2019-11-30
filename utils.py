import sklearn.svm
from grakel import GraphKernel

kernel = GraphKernel

MODEL_MAP = {
    "angular": sklearn.svm.SVC(kernel=angular_kernel),
}