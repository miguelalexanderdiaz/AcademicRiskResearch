# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:12:47 2015

@author: monica pineda,miguel diaz
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.decomposition import PCA
from sklearn.lda import LDA
from numpy import genfromtxt


dimension_number=3


dataset=genfromtxt('data_1.csv', delimiter=';')
X=dataset[:,0:16]
Y=dataset[:,17]


pca = PCA(dimension_number) 
aux=pca.fit(X)
X_r = pca.fit(X).transform(X)

lda = LDA(dimension_number)
aux=lda.fit(X, Y)

X_r2 = lda.fit(X, Y).transform(X)
print X_r2



print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
      

ax=Axes3D(plt.figure())

#ax.plot(xs=x, ys=[0]*len(x), zs=z, zdir='z', label='ys=0, zdir=z')




for c, i, target_name in zip("rgb", [0, 1, 2], ["low","mid","high"]):
    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], X_r[Y==i,2], c=c, label=target_name)
plt.legend()
plt.title('PCA of IRIS dataset')
plt.figure()

for c, i, target_name in zip("rgb", [0, 1, 2], ["low","mid","high"]):
    plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of IRIS dataset')

plt.show()

