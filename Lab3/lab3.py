import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AR
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
    amin = np.array(list(df.min())[:11])
    amax = np.array(list(df.max())[:11])
    v = df.values[:,:11]
    v = ((v-amin)/(amax - amin))*(h-l) + l
    mmdf = pd.DataFrame(v , columns = columns[:11])
    mmdf["quality"] = df.quality
    return mmdf

def standardize(df):
    columns = df.columns
    mean = np.array(list(df.mean())[:11])
    std = np.array(list(df.std())[:11])
    v = df.values[:,:11]
    v = ((v-mean)/(std))
    mmdf = pd.DataFrame(v , columns = columns[:11])
    mmdf["quality"] = df.quality
    print("mean before standardize",df.mean(),"\n\n std before standardize",df.std())
    print("\n\nmean after standardize",mmdf.mean(),"\n\n std after standardize",mmdf.std())
    return mmdf

def replace_outliers(df):
    columns = df.columns
    for i in range(11):
        a = np.array(df[columns[i]])
        a = np.sort(a)
        q1 = a[400]
        q3 = a[1200]
        itr = q3 - q1
        l = q1 - 1.5*itr
        h = q3 + 1.5*itr
        m = a[800]
        df.loc[df[columns[i]] > h , [columns[i]]] = m
        df.loc[df[columns[i]] < l , [columns[i]]] = m
    

def main():
    path_to_file = "winequality_red_original.csv"
    df = read_data(path_to_file)
    columns = df.columns
    
    show_boxplot(columns[:11],df)
    
    replace_outliers(df)
    
    show_boxplot(columns[:11],df)
    
    mmdf = min_max_normalisation(df,0,1)
    
    mmdf = standardize(df)
    
    scaler = MinMaxScaler()
    mmdf = scaler.fit_transform(df[columns[:11]])
    mmdf = pd.DataFrame(mmdf , columns = columns[:11])
    mmdf["quality"] = df.quality
    
    scaler = StandardScaler()
    mmdf = scaler.fit_transform(df[columns[:11]])
    mmdf = pd.DataFrame(mmdf , columns = columns[:11])
    mmdf["quality"] = df.quality

    
if __name__=="__main__":
    main()
    