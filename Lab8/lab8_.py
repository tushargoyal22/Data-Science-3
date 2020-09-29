import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
import scipy

def split(dfn):
    X_train, X_test, y_train, y_test = train_test_split(dfn[df.columns[:len(df.columns)-1]], dfn[df.columns[len(df.columns)-1]] , test_size = 0.3, random_state = 0) 
    return X_train, X_test, y_train, y_test


df = pd.read_excel("AirQuality.xlsx")
df.loc[df["CO"]==-200 , "CO"] = np.nan
df.loc[df["CO"].isnull() , "CO"] = df["CO"].median()

X_train, X_test, y_train, y_test = split(df)

train = X_train.copy()
train["quality"] = y_train
train.to_csv('AirQuality-train.csv') 

test = X_test.copy()
test["quality"] = y_test
test.to_csv('AirQuality-test.csv') 







att1 = "PT08.S1(CO)"
plt.scatter(X_train["PT08.S1(CO)"],y_train)
xt = np.linspace( X_train[att1].mean()-4*X_train[att1].std() , X_train[att1].mean()+4*X_train[att1].std() , 10)
xtp = np.reshape(xt, (len(xt), 1))
x=np.reshape(np.array(X_train["PT08.S1(CO)"]), (len(np.array(X_train["PT08.S1(CO)"])), 1))
reg = linear_model.LinearRegression() 
reg.fit(x,y_train)
y_pred = reg.predict(xtp)
plt.plot(xt,y_pred,c="red")
plt.xlabel("PT08.S1(CO) value")
plt.ylabel("CO")
plt.title("best fit line")
plt.show()


xtr=np.reshape(np.array(X_train["PT08.S1(CO)"]), (len(np.array(X_train["PT08.S1(CO)"])), 1))
xts=np.reshape(np.array(X_test["PT08.S1(CO)"]), (len(np.array(X_test["PT08.S1(CO)"])), 1))
reg = linear_model.LinearRegression() 
reg.fit(xtr, y_train) 
train_pred = reg.predict(xtr)
test_pred = reg.predict(xts)


print("rmse (simple lr) for train data =",(np.sum((train_pred-y_train)**(2))/len(y_train))**(0.5))
print("rmse (simple lr) for test data =",(np.sum((test_pred-y_test)**(2))/len(y_test))**(0.5))


plt.scatter(y_test,test_pred)
plt.xlabel("actual CO")
plt.ylabel("predicted CO")
plt.title("scatter plot on test data predictions (simple lr)")
plt.show()











plt.scatter(X_train["PT08.S1(CO)"],y_train)
att1 = "PT08.S1(CO)"
xt = np.linspace( X_train[att1].mean()-4*X_train[att1].std() , X_train[att1].mean()+4*X_train[att1].std() , 10)
xt=np.reshape(xt, (len(xt), 1))
polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)
xtp = polynomial_features.fit_transform(xt)
regressor = linear_model.LinearRegression()
regressor.fit(x_poly, y_train)
y_pred = regressor.predict(xtp)
plt.plot(xt,y_pred,c="red")
plt.xlabel("PT08.S1(CO) value")
plt.ylabel("CO")
plt.title("best fit curve")
plt.show()



print("\nfor traning data (simple lr)\n\n")
xtr=np.reshape(np.array(X_train["PT08.S1(CO)"]), (len(np.array(X_train["PT08.S1(CO)"])), 1))
rmse=[]
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(xtr)
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_train)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_train)**(2))/len(y_train))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])
    
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("rmse value")
plt.title("for traning data")
plt.show()
print("\nrmse is almost same at every value of p\n")


print("\n\nfor test data (simple lr)\n")
xts=np.reshape(np.array(X_test["PT08.S1(CO)"]), (len(np.array(X_test["PT08.S1(CO)"])), 1))
rmse=[]
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(xts)
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_test)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_test)**(2))/len(y_test))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])
    
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("rmse value")
plt.title("for test data")
plt.show()

print("\nrmse is almost same at every value of p\n")




polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(xts)
regressor = linear_model.LinearRegression()
regressor.fit(x_poly, y_test)
y_pred = regressor.predict(x_poly)
plt.scatter(y_test,y_pred)
plt.xlabel("actual CO")
plt.ylabel("predicted CO")
plt.title("scatter plot on test data predictions when p = 4")
plt.show()



    



















columns = X_train.columns
pcorrcoef = []
for i in columns:
    pcorrcoef.append([i,scipy.stats.pearsonr(X_train[i],y_train)[0]])
print(pcorrcoef,"\nNOx(GT) attribute  and PT08.S5(O3) is most correlated with quality\n")

att1 = "NOx(GT)"
att2 = "PT08.S5(O3)"
att=[att1,att2]
reg = linear_model.LinearRegression() 
reg.fit(X_train[[att1,att2]], y_train)
coef =  reg.coef_
incpt = reg.intercept_ 
a,b,c,d=coef[0],coef[1],-1,-incpt
x = np.linspace( X_train[att1].mean()-2*X_train[att1].std() , X_train[att1].mean()+2*X_train[att1].std() , 10)
y = np.linspace( X_train[att2].mean()-2*X_train[att2].std() , X_train[att2].mean()+2*X_train[att2].std() , 10)
X,Y = np.meshgrid(x,y)
Z = (d - a*X - b*Y) / c
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.scatter(X_train[att1][:400] ,X_train[att2][:400], y_train[:400])
ax.plot_surface(X, Y, Z)
ax.set_xlabel(att1)
ax.set_ylabel(att2)
ax.set_zlabel('quality')
plt.show()



reg = linear_model.LinearRegression() 
reg.fit(X_train[att], y_train) 
train_pred = reg.predict(X_train[att])
test_pred = reg.predict(X_test[att])
print("rmse (two most correlated attribute simple lr) for train data =",(np.sum((train_pred-y_train)**(2))/len(y_train))**(0.5))
print("rmse (two most correlated attribute simple lr) for test data =",(np.sum((test_pred-y_test)**(2))/len(y_test))**(0.5))

plt.scatter(y_test,test_pred)
plt.xlabel("actual CO")
plt.ylabel("predicted CO")
plt.title("scatter plot on test data predictions (two most correlated attribute simple lr)")
plt.show()










print("\nfor traning data (two most correlated attribute multiple lr)\n\n")
rmse=[]
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(X_train[att])
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_train)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_train)**(2))/len(y_train))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])
    
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("rmse value")
plt.title("for traning data")
plt.show()


print("\n\nfor test data (two most correlated attribute multiple lr)\n")
rmse=[]
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(X_test[att])
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_test)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_test)**(2))/len(y_test))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])
    
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("rmse value")
plt.title("for test data")
plt.show()

print("\nrmse is almost same at every value of p\n")




polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(X_test[att])
regressor = linear_model.LinearRegression()
regressor.fit(x_poly, y_test)
y_pred = regressor.predict(x_poly)
plt.scatter(y_test,y_pred)
plt.xlabel("actual CO")
plt.ylabel("predicted CO")
plt.title("scatter plot on test data predictions when p = 4 (two most correlated attribute multiple lr)")
plt.show()




polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(X_train[att])
regressor = linear_model.LinearRegression()
regressor.fit(x_poly, y_train)

coef =  regressor.coef_
incpt = regressor.intercept_ 
xc = np.linspace( X_train[att1].mean()-2*X_train[att1].std() , X_train[att1].mean()+2*X_train[att1].std() , 10)
yc = np.linspace( X_train[att2].mean()-2*X_train[att2].std() , X_train[att2].mean()+2*X_train[att2].std() , 10)
X,Y = np.meshgrid(xc,yc)
Xi = np.array([X.ravel().tolist(),Y.ravel().tolist()])
polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(Xi.T)
Z=np.sum(x_poly*coef,axis=1)+incpt
Z = Z.reshape(X.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z,color="red")
ax.scatter(X_train[att1][:500] ,X_train[att2][:500], y_train[:500])
ax.set_xlabel(att1)
ax.set_ylabel(att2)
ax.set_zlabel('CO')
plt.show()
