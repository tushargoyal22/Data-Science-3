# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:05:14 2019

@author: hp
"""
#%%

import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

#%%

df = pd.read_csv(r'H:\Aug-Dec 2019\IC 272\Labs\Lab1\winequality-red.csv', sep = ';')

#%%

df2 = df.agg(['min', 'mean', 'max', 'median', 'std'])
print(df2)
print()

#%%

for attribute in df.columns:
    if attribute != "quality":
        plt.scatter(df.quality, df[attribute], )
        plt.xlabel('quality')
        plt.ylabel(attribute)
        print("Pearson coefficient:", st.pearsonr(df.quality, df[attribute])[0])
        plt.show()
print()

#%%

for column in df.columns:
    plt.hist(df[column])
    plt.show()
print()

#%%

df3 = df.groupby('quality')
for i, j in df3:
    print(j.loc[:, ['pH', 'quality']])
    plt.hist(df['pH'])
    plt.show()
print()

#%%

'''
df.loc[:, :"residual sugar"].boxplot()
plt.show()
df.loc[:, "chlorides":"density"].boxplot()
plt.show()
df.loc[:, "pH":"quality"].boxplot()
plt.show()
'''

for column in df.columns:
    df.boxplot(column = [column])
    plt.show()
    
#%%

