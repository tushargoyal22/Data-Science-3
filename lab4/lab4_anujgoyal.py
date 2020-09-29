# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 04:15:18 2019

@author: anujg
"""
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

import numpy as np
import matplotlib.pyplot as plt



X=[]
Y=[]
for i in range(1000):
    x, y = np.random.multivariate_normal([0,0],[[7,10],[10,18]],1).T
    X=X+list(x)
    Y=Y+list(y)

C=[]
for i in range(len(X)):
    C.append([X[i],Y[i]])
C=np.array(C)

plt.scatter(C.T[0],C.T[1],marker='x')

# calculate covariance matrix of centered matrix
V = cov(C.T)
print(V)

# eigen decomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)
qt=vectors.transpose()
plt.quiver(qt[0][0],qt[0][1],scale=5,color='r')
plt.quiver(qt[1][0],qt[1][1],scale=5,color='r')

# project data
P = vectors.T.dot(C.T)     #project on both eigen vectors
D1=vectors.T[0].dot(C.T)   #project on eigen vector 1
D2=vectors.T[1].dot(C.T)   #project on eigen vector 2

#reconstructing the data
D0p=[]     #from both vectors
D1p=[]     #from eigen vector 1
D2p=[]     #from eigen vector 2
for i in range(1000):
    D1p.append(D1[i]*vectors.T[0])
    D2p.append(D2[i]*vectors.T[1])
    D0p.append(np.add(P.T[i][0]*vectors.T[0],P.T[i][1]*vectors.T[1]))

D1p=np.array(D1p).transpose()
D2p=np.array(D2p).transpose()
  
#projecting data in one dimension
plt.scatter(D1p[0],D1p[1],marker='x')    #along vector 1
plt.scatter(D2p[0],D2p[1],marker='x')    #along vector 2


plt.xlim([-25,20])
plt.ylim([-15,15])



#finding error in reconstructed data and original data
E1=[]
E2=[]
E0=[]

#finding the eucledian distance
for i in range(1000):
    e0=((C[i][0]-D0p[i][0])**2+(C[i][1]-D0p[i][1])**2)
    e1=((C[i][0]-D1p.T[i][0])**2+(C[i][1]-D1p.T[i][1])**2)
    e2=((C[i][0]-D2p.T[i][0])**2+(C[i][1]-D2p.T[i][1])**2)     
    E0.append(e0)
    E1.append(e1)    
    E2.append(e2)

error0=sum(E0)**0.5  #error when both vectors are used 
error1=sum(E1)**0.5  #error when vector 1 is used
error2=sum(E2)**0.5  #error when vector 2 is used  


#saving original and reduced data
np.savetxt("original.csv",C , delimiter=",") #original data
np.savetxt("D1.csv",D1, delimiter=",")       #data from vector 1
np.savetxt("D2.csv",D2, delimiter=",")       #data from vector 2
    