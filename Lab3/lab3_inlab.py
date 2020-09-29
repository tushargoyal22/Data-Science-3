#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:06:47 2019

@author: tushar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def read_data(path_to_file):
    df = pd.read_csv(path_to_file)
    return df

def show_boxplot(columns,df):
    for i in columns:
        plt.boxplot(df[i])
        plt.title("boxplot for "+i)
        plt.show()

def rnge(df,columns):
    for i in columns:
        print("min and max value of attribute ",i," are ",df[i].min()," and ",df[i].max())
    
def min_max_normalisation(df,l,h):
    columns = df.columns
    amin = np.array(list(df.min()))
    amax = np.array(list(df.max()))
    v = df.values
    
    v = ((v-amin)/(amax - amin))*(h-l) + l
    mmdf = pd.DataFrame(v , columns = columns)
#    mmdf["quality"] = df.quality
    return mmdf

def standardize(df):
    columns = df.columns
    mean = np.array(list(df.mean()))
    std = np.array(list(df.std()))
    v = df.values
    v = ((v-mean)/(std))
    mmdf = pd.DataFrame(v , columns = columns)
    mmdf["quality"] = df.quality
    print("mean before standardize",df.mean(),"\n\n std before standardize",df.std())
    print("\n\nmean after standardize",mmdf.mean(),"\n\n std after standardize",mmdf.std())
    return mmdf

def replace_outliers(df):
    columns = df.columns
    x=df.quantile(0.25)
    y=df.quantile(0.75)
    z=df.quantile(0.50)
    x=list(x)
    y=list(y)    
    z=list(z)
    for i in range(0,3):
        q1 = x[i]
        q3 = y[i]
        itr = q3 - q1
        l = q1 - 1.5*itr
        h = q3 + 1.5*itr
        m = z[i]
        df.loc[df[columns[i]] > h , [columns[i]]] = m
        df.loc[df[columns[i]] < l , [columns[i]]] = m
        
    

def main():
    path_to_file = "landslide_data2_original.csv"
    df = read_data(path_to_file)
    x=df.values[:,5:]
    df=pd.DataFrame(x,columns=["temperature","humidity","rain"])
    columns = df.columns
    
    show_boxplot(columns,df)
 
    replace_outliers(df)
    
    show_boxplot(columns,df)
    
    mmdf = min_max_normalisation(df,0,1)
#    mmdf2 = min_max_normalisation(df,0,20)
    #print(mmdf,mmdf2)
    
    mmdf = standardize(df)
    
    scaler = MinMaxScaler()
    mmdf = scaler.fit_transform(df[columns])
    mmdf = pd.DataFrame(mmdf , columns = columns)
#    mmdf["quality"] = df.quality
    print(mmdf)
    scaler = StandardScaler()
    mmdf = scaler.fit_transform(df[columns])
    mmdf = pd.DataFrame(mmdf , columns = columns)
#    mmdf["quality"] = df.quality
    print(mmdf)
    
if __name__=="__main__":
    main()
    
