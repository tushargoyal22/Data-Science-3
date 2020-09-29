from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import homogeneity_score as hs
from sklearn.mixture import gaussian_mixture as GMM
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:09:15 2019

@author: abhimanhas
"""
from sklearn import metrics
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) 


f=open("2D_points2.txt","r")
p1=[]
p2=[]
for i in f.readlines():
    tmp=i.split()
    p1.append(float(tmp[0]))
    p2.append(float(tmp[1]))
X=pd.DataFrame({'x':p1,'y':p2})    
y=[]
for i in range(2000):
    if i<500:
        y.append(0)
    elif i<1000:
        y.append(1)
    elif i<1500:
        y.append(2)
    else:
        y.append(3)
        
        
k=[2,3,4,5,10]
kmeans=KMeans(n_clusters=4,random_state=0).fit(X)
centers=kmeans.cluster_centers_
labels=kmeans.labels_
y_kmeans = kmeans.predict(X)
plt.scatter(p1,p2, c=y_kmeans, s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='black',s=200)
plt.show()
print("sum of sq dist = ",kmeans.inertia_)
print("purity : ",purity_score(y,y_kmeans))
print("homogenousity score: ",hs(y,y_kmeans))
vals=[]
for i in k:
    km=KMeans(n_clusters=i,random_state=0).fit(X)
    vals.append(km.inertia_)
plt.plot(k,vals)
plt.show()
kmedoids = KMedoids(n_clusters=4, random_state=0).fit(X)
centers=kmedoids.cluster_centers_
labels=kmedoids.labels_
y_kmed=kmedoids.predict(X)
y_kmeans = kmedoids.predict(X)
plt.scatter(p1,p2, c=y_kmeans, s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='black',s=200)
plt.show()
print("sum of sq dist = ",kmedoids.inertia_)
print("purity : ",purity_score(y,y_kmed))
print("homegenousity score : ",hs(y,y_kmed))
vals=[]
for i in k:
    km=KMedoids(n_clusters=i,random_state=0).fit(X)
    vals.append(km.inertia_)
plt.plot(k,vals)
plt.show()

gmm = GMM(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(p1,p2, c=labels, s=40, cmap='viridis')