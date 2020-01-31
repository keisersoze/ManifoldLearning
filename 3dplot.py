from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from Z_points_generator import spk_isomap
from datasets_utils import load_shock_dataset
from datasets_utils import load_ppi_dataset

# Parameters
#KNN
min_x = 5
max_x = 50
#D
min_y = 5
max_y = 50
#K-fold
k_fold = 10

svmC = 100

#Data selection
X, y = load_shock_dataset()
#X, y = load_ppi_dataset()

# Shuffle data
idx = np.random.RandomState(seed=42).permutation(len(X))
X_data, y_data = X[idx], y[idx]

x = np.arange(min_x,max_x+1,1)
y = np.arange(min_y,max_y+1,1)
X,Y = np.meshgrid(x,y)

Z = spk_isomap(X_data,y_data,k_fold,min_x,max_x,min_y,max_y,svmC)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')

mycmap = plt.get_cmap('gist_earth')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.6, cmap=mycmap)
cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=mycmap)
# cset = ax.contourf(X, Y, Z, zdir='x', offset=-5, cmap=mycmap)
# cset = ax.contourf(X, Y, Z, zdir='y', offset=5, cmap=mycmap)


fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


ax.set_xlabel('k')
ax.set_xlim(min_x, max_x+1)
ax.set_ylabel('d')
ax.set_ylim(min_y, max_y+1)
ax.set_zlabel('Avg. accuracy')
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_title('Avg. Accuracy 3D and contour plot over parameters k and d')


plt.show()