from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from random import random

x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)
X,Y = np.meshgrid(x,y)
Z = np.zeros((len(y), len(x)))
for i, _ in enumerate(y):
    for j, _ in enumerate(x):
        Z[i, j] = random() * 0.5


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')

mycmap = plt.get_cmap('gist_earth')
surf = ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.8, cmap=mycmap)
cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=mycmap)
#cset = ax.contourf(X, Y, Z, zdir='x', offset=-5, cmap=mycmap)
#cset = ax.contourf(X, Y, Z, zdir='y', offset=5, cmap=mycmap)


fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


ax.set_xlabel('X')
ax.set_xlim(-5, 5)
ax.set_ylabel('Y')
ax.set_ylim(-5, 5)
ax.set_zlabel('Z')
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_title('3D surface with 2D contour plot projections')


plt.show()