#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:00:49 2019

@author: user
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def read_data(path_to_file):
    df=pd.read_csv(path_to_file)
    return df
def show_box_plot(attribute_name,dataframe):
    plt.boxplot(dataframe[attribute_name])
    plt.title("Boxplot for attribute "+attribute_name)
    plt.show()
def replace_outliers(dataframe,attribute):
    columns = list(dataframe.columns)
    x=dataframe[attribute].quantile(0.25)
    y=dataframe[attribute].quantile(0.75)
    z=dataframe[attribute].quantile(0.50)
    
    
    q1 = x        
    q3 = y
    itr = q3 - q1
    l = q1 - 1.5*itr
    h = q3 + 1.5*itr
    m = z
    dataframe.loc[dataframe[attribute] > h , [attribute]] = m
    dataframe.loc[dataframe[attribute] < l , [attribute]] = m
        

def min_max_normalization(df,h,l):
    columns = list(df.columns)[5:]
    amin = np.array(list(df.min())[5:])
    amax = np.array(list(df.max())[5:])
    v = df.values[:,5:]
    
    v = ((v-amin)/(amax - amin))*(h-l) + l
    mmdf = pd.DataFrame(v , columns = columns[:])
    
    return mmdf

def standardize(df,attribute):
    
    mean =df[attribute].mean()
    std = df[attribute].std()
    v = df[attribute].values
    v = ((v-mean)/(std))
    
    
    print("mean before standardize",df[attribute].mean(),"\n\n std before standardize",df[attribute].std())
    print("\n\nmean after standardize",v.mean(),"\n\n std after standardize",v.std())
    

def main():
	
    path_to_file="/local/user/Downloads/landslide_data2_original(2).csv"
    df=read_data(path_to_file)
    print("Before removing outliers:\n")
    show_box_plot('temperature',df)
    show_box_plot('humidity',df)
    show_box_plot('rain',df)
    replace_outliers(df,'temperature')
    replace_outliers(df,'humidity')
    replace_outliers(df,'rain')
    print("After removing outliers:\n")
    show_box_plot('temperature',df)
    show_box_plot('humidity',df)
    show_box_plot('rain',df)
    print(min_max_normalization(df,1,0))
    print(min_max_normalization(df,0,20))
    standardize(df,'humidity')
    standardize(df,'rain')
    standardize(df,'temperature')

if __name__=="__main__":
	main()
