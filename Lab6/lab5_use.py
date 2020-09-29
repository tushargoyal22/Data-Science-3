# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:48:26 2019

@author: lmd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


global_accuracy = []

def load_dataset(path_to_file):
    df = pd.read_csv(path_to_file)
    return df


def normalization(df):
    feature_df = df.drop('class', axis=1)

    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(feature_df), columns = df.columns[0:-1])
    df_norm['class'] = df['class']
    df_norm.to_csv('pima-indians-diabetes-Normalised.csv')

    return df_norm


def standardize(df):
    feature_df = df.drop('class', axis=1)
    scaler = StandardScaler()
    df_standard = pd.DataFrame(scaler.fit_transform(feature_df), columns = df.columns[0:-1])
    df_standard['class'] = df['class']
    df_standard.to_csv('pima-indians-diabetes-Standardised.csv')

    return df_standard


def traintestsplit(df, save_train, save_test):
    X = df.iloc[:, 0:-1].values
    y = df['class'].values

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,stratify=y, random_state=42,shuffle=True)

    feature_columns = list(df.columns[0:-1])

    train_data = pd.DataFrame(X_train, columns = feature_columns)
    train_data['class'] = y_train
    train_data.to_csv(save_train)

    test_data = pd.DataFrame(X_test, columns = feature_columns)
    test_data['class'] = y_test
    test_data.to_csv(save_test)

    return X_train, X_test, y_train, y_test


def classification(df, save_train, save_test):
    X_train, X_test, y_train, y_test = traintestsplit(df, save_train, save_test)
    neighbors = list(range(1, 25, 2))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

        y_pred = knn.predict(X_test)
        print("Accuracy for k = ", k, ":", test_accuracy[i], '\n')
        print("Confusion matrix for k = ", k, '\n', confusion_matrix(y_test, y_pred))
        print('\n')

    print('\n')

    global_accuracy.append(test_accuracy)

    print("Value of K with highest accuracy:", 2*(np.argmax(test_accuracy)) + 1)
    print("Value of K at this accuracy:", np.max(test_accuracy))
    print("\n")


    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()


def main():

    df = load_dataset('pima-indians-diabetes.csv')
    df_norm = normalization(df)
    df_standard = standardize(df)

    print("Original data\n\n")
    classification(df, 'pima-indians-diabetes-train.csv', 'pima-indians-diabetes-test.csv')
    print("\n\n***************************\n\n")


    print("Normalized data\n\n")
    classification(df_norm, 'pima-indians-diabetes-normalise.csv', 'pima-indians-diabetes-test-normalise.csv')
    print("\n\n***************************\n\n")


    print("Standardise data\n\n")
    classification(df_standard, 'pima-indians-diabetes-train-standardise.csv', 'pima-indians-diabetes-test-standardise.csv')


    x = list(range(1, 25, 2))
    plt.plot(x, global_accuracy[0], label = "Original data")
    plt.plot(x, global_accuracy[1], label = "Normalized data")
    plt.plot(x, global_accuracy[2], label = "Standardized data")
    plt.legend()
    plt.xlabel("Number of neighbors",fontweight='bold')
    plt.ylabel("Accuracy")
    plt.savefig('Accuracy_vs_k_on_original_normalised_standardised.png')
    plt.show()
#    plt.savefig('Accuracy_vs_k_on_original_normalised_standardised.png')


main()
