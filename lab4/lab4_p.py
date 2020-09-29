# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:20:54 2019
@author: 
"""

import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof

def meanmat():return[int(i) for i in input("Enter the means:").split()]

def covmat():return[[int(input("Enter c"+str(i)+str(j)+": ")) for j in range(2)] for i in range(2)]

def dimension(data): return data.shape[1]

def printsize(data):print("\nSize of 1000 data points:",getsizeof(data))

def arb2x1mat():return[int(i) for i in input("Enter the 2x1 matrix A for transformation:").split()]

def mserror(data,transformed_data):return sum(list((sum([((data-transformed_data).T[i])**2 for i in range(dimension(data))]))))

def printerror(data,transformed_data):print("Error in the transformation is:",mserror(data,transformed_data))

def generate_data(mean,cov,size): x,y = np.random.multivariate_normal(mean,cov,size).T; return np.array([x,y]).T

def plot2D(N): plt.scatter(N.T[0],N.T[1])



mean = meanmat()
cov = covmat()

sample_data = generate_data(mean,cov,1000)

plot2D(sample_data.copy())
plt.axis('scaled')
plt.show()

printsize(sample_data.copy())

#Inferences:
#Scatter plot is centered at (mean1,mean2) and dispersed according to the cov matrix.
#Eigenvectors would be along the directions in which there is max dispersion of data.

A = np.array([[1,2]]).T
#A = np.array([arb2x1mat()]).T

sample_data_prime = np.matmul(sample_data.copy(),A)
print()
print("Error in this transformation:",mserror(sample_data.copy(),sample_data_prime))
print()

#Inferences:
#For different matrices Ax, the reconstruction error would vary.
#This would be minimum if A is chosen in a direction parallel to the maximum dispersion of datapoints.
#i.e. A should be an eigenvector. Also, it should be the eigenvector corresponding to max eigenvalue to be
#in direction of max. dispersion.

C = np.matmul(sample_data.copy().T,sample_data.copy())
print("Original Covariance Matrix:\n",C,"\n")

evals,evecs = np.linalg.eig(C)
projection_evecs = [evecs.T[i].T for i in range(dimension(sample_data))]

all_eigen_transform = np.matmul(sample_data.copy(),evecs)
print("Applying transformation of matrix with all Eigenvectors..")


C_transform = np.matmul(all_eigen_transform.T,all_eigen_transform)
print("Eigenvalues:",evals,"\n")
print("Covariance Matrix after transform:\n",C_transform,"\n")

plot2D(sample_data.copy())
for i in range(sample_data.shape[1]): plt.quiver([1],projection_evecs[i][1]/projection_evecs[i][0],scale_units='xy',scale = 0.5)
plt.axis('scaled')
plt.show()
printerror(sample_data.copy(),all_eigen_transform)

#Inferences:
#As it is clear that eigenvalues are same as the variances in the covariance matrix of transformed data.
#Also, for dimensional reduction, we would use the eigen vector parallel to the direction of max dispersion.
#That is the eigenvector corresponding to the larger eigenvalue.

projections = [np.matmul(sample_data.copy(),projection_evecs[i]) for i in range(dimension(sample_data))]

projected_data = [np.array([projections[i]*projection_evecs[i][0],projections[i]*projection_evecs[i][1]]).T for i in range(dimension(sample_data))]

for i in range(dimension(sample_data)):
    plot2D(sample_data.copy())
    for j in range(sample_data.shape[1]): plt.quiver([1],projection_evecs[j][1]/projection_evecs[j][0],scale_units='xy',scale = 0.5)
    plt.scatter(projected_data[i].T[0],projected_data[i].T[1])
    plt.axis('scaled')
    plt.show()
    printerror(sample_data.copy(),projected_data[i])
    
#Inferences:
#Error greatly decreases in the eigenvector transformation corresponding to largest eigenvelue.
