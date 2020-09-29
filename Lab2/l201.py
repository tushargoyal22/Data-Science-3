import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

odf = pd.read_csv("winequality-red_original .csv",delimiter=";")
codf=odf.copy()

df = pd.read_csv("winequality-red_miss.csv")
print("nos. of NaN acc. to attributes: ")
print(df.isnull().sum())
print("\ntotal no. of null values : ",df.isnull().sum().sum(),"\n")

p=[]
a = np.zeros(13)
for t in range(0,1599):
    c=df.iloc[[t]].isnull().sum().sum()
    a[c]=a[c]+1
    if (c>=6):
        p.append(t)
m=0
for i in range(1,13):
    print("Nos. of tuples having ",i," missing values : ",int(a[i]))
    if (i>=6):
        m=m+int(a[i])
        
plt.bar(range(1,13),a[1:])
plt.xlabel("No. of missing values.")
plt.ylabel("No. of tuples.")
plt.show()
print("No.of tuples having equal to or more than 50% of attributes with missing values : ",m)
df.drop(df.index[p],inplace=True)
codf.drop(df.index[p],inplace=True)
df = df[pd.notnull(df['quality'])]
codf = codf[df.index[pd.notnull(df['quality']).index]]
print("\n\nnos. of NaN acc. to attributes: ")
print(df.isnull().sum())
print("\ntotal no. of null values after deleting tuples: ",df.isnull().sum().sum(),"\n")





df1=df.copy()
df1.fillna(df1.median(),inplace=True)
att=df.columns



m1 = pd.DataFrame()
m1["attributes"]=att
m1["orignal data"]=odf.mean(axis=0).values
m1["missing values"]=df1.mean(axis=0).values

print("\n\n\n\nMEAN VALUES when missing values filled with median values")
print(m1)

#for i in att:
#    print("\n Boxplot for attribute",i)
#    plt.boxplot([df1[i],odf[i]])
#    plt.xlabel("Data with filled missing values                 original data")
#    plt.show()

x=np.argwhere(np.isnan(df.values))
print(x)
sse=0
for i in x:
    sse=sse+(codf.values[i[0]][i[1]]-df1.values[i[0]][i[1]])*(codf.values[i[0]][i[1]]-df1.values[i[0]][i[1]])
print("\n\n\n\nroot mean square error (RMSE) between the original and replaced values when filled with median value: ",sse)






df2=df.copy()
df2.fillna(method="pad",inplace=True)

m2 = pd.DataFrame()
m2["attributes"]=att
m2["orignal data"]=odf.mean(axis=0).values
m2["missing values"]=df2.mean(axis=0).values

print("\n\nMEAN VALUES when missing values filled with preceding values")
print(m2)

#for i in att:
#    print("\n Boxplot for attribute",i)
#    plt.boxplot([df2[i],odf[i]])
#    plt.xlabel("Data with filled missing values                 original data")
#    plt.show()

sse=0

print(y)
for i in x:
    if i not in y:
        sse=sse+(codf.values[i[0]][i[1]]-df2.values[i[0]][i[1]])*(codf.values[i[0]][i[1]]-df2.values[i[0]][i[1]])
print("\n\n\n\n\nroot mean square error (RMSE) between the original and replaced values when filled with preceding values : ",sse)






df3=df.copy()
df3.interpolate(method="linear",inplace=True)

m3 = pd.DataFrame()
m3["attributes"]=att
m3["orignal data"]=odf.mean(axis=0).values
m3["missing values"]=df3.mean(axis=0).values

print("\n\nMEAN VALUES when missing values filled by linear interpolating")
print(m3)

#for i in att:
#    print("\n Boxplot for attribute",i)
#    plt.boxplot([df3[i],odf[i]])
#    plt.xlabel("Data with filled missing values                 original data")
#    plt.show()

sse=0
for i in x:
    if i not in y:
        sse=sse+(codf.values[i[0]][i[1]]-df3.values[i[0]][i[1]])*(codf.values[i[0]][i[1]]-df3.values[i[0]][i[1]])
print("\n\n\n\nroot mean square error (RMSE) between the original and replaced values when filled by linear interpolating : ",sse)

