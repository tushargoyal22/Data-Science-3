#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 21:03:57 2019

@author: spider
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
# Import or Load the data
def load_dataset(path_to_file):
    df=pd.read_csv(path_to_file)
    return df
###################################################################################

# Data Preprocessing (Use only the required functions for the assignment)
"""
- Standardization/Normalization
- Train/Test Split
"""
###################################################################################
def Normalize(df,end,minm=0,maxm=1):
  X=df.loc[:,df.columns!=end]
  X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
  X_scaled = X_std*(maxm - minm) + minm
  X_scaled[end]=df[end]
  return X_scaled

def Standarize(dataframe,end):
  X=dataframe.loc[:,df.columns!=end]
  X_scaled=(X-X.mean())/X.std()
  X_scaled[end]=dataframe[end]
  return X_scaled
# Saving the data
def save_file(dataframe,location):
  dataframe.to_csv(location)
#knn analysis
def knn_analysis(dataframe):
  
  X=dataframe
  X_label=dataframe["Class"]
  X_train, X_test, X_label_train, X_label_test =train_test_split(X, X_label, test_size=0.3, random_state=42,stratify=X_label,shuffle=True)
  accuracy=[]
  save_file(X_train,"DiabeticRetinipathy-train.csv")
  save_file(X_test,"DiabeticRetinipathy-text.csv")
  klist=[1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
  for k in klist:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, X_label_train)
    y_pred = knn.predict(X_test)
    print("Confusion Matrix for k=",k,"\n",confusion_matrix(X_label_test, y_pred))
    print("Accuracy for k=:",k,"  ",metrics.accuracy_score(X_label_test, y_pred))
    accuracy.append(metrics.accuracy_score(X_label_test, y_pred))
  
  plt.plot(klist,accuracy)
  plt.xlabel("K Values")
  plt.ylabel("Accuracy")
  plt.show()
  return accuracy
# =============================================================================
# Write a python program to
# a. Normalize all the attributes, except class attribute, of DiabeticRetinipathy.csv
# using min-max normalization to transform the data in the range [0-1]. Save the file as
# DiabeticRetinipathy-Normalised.csv
# b. Standardize, all the attributes, except class attribute, of DiabeticRetinipathy.csv
# using
# z-normalization.
# Save
# the
# file
# as
# DiabeticRetinipathy-
# Standardised.csv
# =============================================================================
klist=[1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
df=load_dataset("DiabeticRetinipathy.csv")
save_file(Standarize(df,"Class"),"DiabeticRetinipathy-Standardised.csv")
save_file(Normalize(df,"Class"),"DiabeticRetinipathy-Normalised.csv")
# =============================================================================
# Split the data of each class from DiabeticRetinipathy.csv into train data and test
# data. Train data contain 70% of tuples from each of the class and test data contain remaining
# 30% of tuples from each class. Save the train data as DiabeticRetinipathy-train.csv
# and save the test data as DiabeticRetinipathy-test.csva. Classify every test tuple using K-nearest neighbor (KNN) method for the different values
# of K (1, 3, 5, 7, 9, 11, 13, 15, 17, 21). Perform the following analysis :
# i. Find confusion matrix (use ‘confusion_matrix’) for each K.
# ii. Find the classification accuracy (You can use ‘accuracy_score’) for each K. Note the
# value of K for which the accuracy is high.
# =============================================================================
print("Output for Q1.")
ao=knn_analysis(df)
# =============================================================================
# Split the data of each class from DiabeticRetinipathy-Normalised.csv into train
# data and test data. Train data should contain same 70% of tuples in Question 2 from each of the
# class and test data contain remaining same 30% of tuples from each class. Save the train data as
# DiabeticRetinipathy-train-normalise.csv and save the test data as
# DiabeticRetinipathy-test-normalise.csv
# a. Classify every test tuple using K-nearest neighbor (KNN) method for the different values
# of K (1, 3, 5, 7, 9, 11, 13, 15, 17, 21). Perform the following analysis :
# i. Find confusion matrix (use ‘confusion_matrix’) for each K.
# ii. Find the classification accuracy (You can use ‘accuracy_score’) for each K. Note the
# value of K for which the accuracy is high.
# =============================================================================
print("Output for Q2.")
an=knn_analysis(Normalize(df,"Class"))
# =============================================================================
# Split the data of each class from DiabeticRetinipathy-Standardised.csv into
# train data and test data. Train data should contain same 70% of tuples in Question 2 from each
# of the class and test data contain remaining same 30% of tuples from each class. Save the train
# data as DiabeticRetinipathy-train-standardise.csv and save the test data as
# DiabeticRetinipathy-test-standardise.csv
# a. Classify every test tuple using K-nearest neighbor (KNN) method for the different values
# of K (1, 3, 5, 7, 9, 11, 13, 15, 17, 21). Perform the following analysis :
# i. Find confusion matrix (use ‘confusion_matrix’) for each K.
# ii. Find the classification accuracy (You can use ‘accuracy_score’) for each K. Note the
# value of K for which the accuracy is high.
# =============================================================================
print("Output for Q3.")
ass=knn_analysis(Standarize(df,"Class"))

# =============================================================================
# 
# Plot and the classification accuracy vs K. for each cases (original, normalized and standardized)
# in a same graph and compare & observe how it is behaving.
# =============================================================================
plt.bar(np.array(klist)-np.array([0.6]*10),np.array(ao), width=0.6, color='b', align='center',label="Original Dataset")
plt.bar( klist,np.array(an), width=0.6, color='g', align='center',label="Normalised Dataset")
plt.bar( np.array(klist)+np.array([0.6]*10),np.array(ass), width=0.6, color='r', align='center',label="Standardised Dataset")
plt.xlabel("K values")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



