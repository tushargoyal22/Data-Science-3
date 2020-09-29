# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#import xlrd

def load_dataset(path_to_file):
    df = pd.read_excel(path_to_file)
    col=df.columns
    for i in col:
        df[i]=df[i].replace(-200,np.NaN)
        
    for i in col:
        df[i]=df[i].replace(np.NaN,df[i].median())
        
    return df


def traintestsplit(df):
    X = df.iloc[:, 0:-1].values
    y = df['CO'].values

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def traintestsplit_l(df):
    X = df.iloc[:,0].values.reshape(-1,1)
    y = df['CO'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)
    return X_train, X_test, y_train, y_test

#load the dataset
df=load_dataset('AirQuality.xlsx')
def linear_reg():
        # Create training and test sets
    print('-'*60)
    print()
    print('Simple Linear Regression')
    X_train, X_test, y_train, y_test = traintestsplit_l(df)
    reg = LinearRegression()# Create the regressor: reg
    # Fit the regressor to the training data
    reg.fit(X_train,y_train)
    prediction_space=np.linspace(min(X_train), max(X_train))
    y_pred_t=reg.predict(prediction_space)#predicted values for training data
    plt.scatter(X_train,y_train,label='scatter plot of training dataset')
    # Plot regression line:#a. Plot the best fit line on the training data where x-axis is pH value and y-axis is quality
    plt.plot(prediction_space, y_pred_t, color='black', linewidth=3,label='Best fit Linear Reg line on the training data')
    plt.xlabel('PT08.S1(CO)')
    plt.ylabel('CO')
    plt.legend()
    plt.show()
    
    # Predict on the test data: y_pred
    y_pred = reg.predict(X_test)
    df_p = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df_p.head(10))
    print('-'*60)
    # Compute and print R^2 and RMSE
    #    print("R^2 Score /Accuracy for train data: {}".format(reg.score(X_train, y_train)))
    #    print("R^2 Score /Accuracy for test data: {}".format(reg.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    print("Root Mean Squared Error: {}".format(rmse))
    print('-'*60)
    print('Scatter plot of actual quality vs predicted quality on the test data')
    plt.scatter(y_test,y_pred)
    plt.xlabel('y_test/Actual CO  =>')
    plt.ylabel('y_pred /Predicted CO =>')
    plt.show()
    print('-'*23+'Q1 Completed'+'-'*24)
linear_reg()

def multi_reg():
    X_train, X_test, y_train, y_test = traintestsplit(df)
    reg = LinearRegression()# Create the regressor: reg
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    # Compute and print R^2 and RMSE
#    print("R^2 Score /Accuracy for train data: {}".format(reg.score(X_train, y_train)))
#    print("R^2 Score /Accuracy for test data: {}".format(reg.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    print("Root Mean Squared Error: {}".format(rmse))
    plt.scatter(y_test,y_pred)
    plt.xlabel('y_test  =>')
    plt.ylabel('y_pred  =>')
    plt.show()

    return y_pred

#multi_reg()

def simple_poly_reg(deg):
    print('-'*60)
    print(' Simple Polynomial Regression of degree',deg,'=>')
    X_train, X_test, y_train, y_test = traintestsplit_l(df)

    polynomial_features= PolynomialFeatures(degree = deg)
    X_train_transform = polynomial_features.fit_transform(X_train)
    X_test_transform = polynomial_features.fit_transform(X_test)
    reg = LinearRegression()# Create the regressor: reg
    reg.fit(X_train_transform,y_train)
    y_pred_on_test = reg.predict(X_test_transform)
    y_pred_on_train = reg.predict(X_train_transform)

    # Compute and print R^2 and RMSE
    print("R^2 Score /Accuracy for train data: {}".format(reg.score(X_train_transform, y_train)))
    print("R^2 Score /Accuracy for test data: {}".format(reg.score(X_test_transform, y_test)))
    rmse_test = np.sqrt(mean_squared_error(y_test,y_pred_on_test))
    rmse_train = np.sqrt(mean_squared_error(y_train,y_pred_on_train))
    print("Root Mean Squared Error on testing data: {}".format(rmse_test))
    print("Root Mean Squared Error on training data: {}".format(rmse_train))
    print()
    print(' '*4+'-'*4+'Best fit curve on the training data'+'-'*4+' '*4)
#    print('Scatter plot of actual quality vs predicted quality on the test data')

#    plt.scatter(y_test,y_pred)
#    plt.xlabel('Actual Quality / y_test  =>')
#    plt.ylabel('Predicted Quality / y_pred  =>')
#    plt.show()

    prediction_space=np.linspace(min(X_train), max(X_train))
    prediction_space_t=polynomial_features.fit_transform(prediction_space.reshape(-1, 1))
    y_pred_predictspace=reg.predict(prediction_space_t)#predicted values for training data
    plt.scatter(X_train,y_train)
    # Plot regression line:#a. Plot the best fit line on the training data where x-axis is pH value and y-axis is quality
    plt.plot(prediction_space, y_pred_predictspace, color='black', linewidth=3,label='Curve of degree {}'.format(deg))
    plt.xlabel('PT08.S1(CO)')
    plt.ylabel('CO')
    plt.legend()
    plt.show()

    return  [rmse_test,rmse_train]

def plots_simple_poly_regression():
    rms_error_test = []
    rms_error_train=[]
    for i in range(2, 7):
        rms_error_test.append(simple_poly_reg(i)[0])
        rms_error_train.append(simple_poly_reg(i)[1])

    print(min(rms_error_train),max(rms_error_train))
    print(min(rms_error_test),min(rms_error_test))
    plt.ylim(0.825,0.83)
    plt.title('Bar graph of RMSE vs different values of degree of polynomial For Test Data')
    plt.bar(list(range(2,7)),rms_error_test)
    plt.xlabel('Degree of Polynomial p=>')
    plt.ylabel('rmse for model with degree p on test data')
    plt.show()

    plt.ylim(0.806,0.81)
    plt.title('Bar graph of RMSE vs different values of degree of polynomial For Train Data')
    plt.bar(list(range(2,7)),rms_error_train)
    plt.xlabel('Degree of Polynomial p=>')
    plt.ylabel('rmse for model with degree p on test data')
    plt.show()

#plots_simple_poly_regression()

def multi_poly_reg(deg):
    print('-'*60)
    print(' Multi Polynomial Regression of degree',deg,'=>')
    X_train, X_test, y_train, y_test = traintestsplit(df)

    polynomial_features= PolynomialFeatures(degree = deg)
    X_train_transform = polynomial_features.fit_transform(X_train)
    X_test_transform = polynomial_features.fit_transform(X_test)
    reg = LinearRegression()# Create the regressor: reg
    reg.fit(X_train_transform,y_train)
    y_pred_on_test = reg.predict(X_test_transform)
    y_pred_on_train = reg.predict(X_train_transform)
#    print('(poly deg ',deg,') linear model coeff (w):\n{}'.format(reg.coef_))
#    print('(poly deg ',deg,')linear model intercept (b): {:.3f}'.format(reg.intercept_))
#    print(reg.coef_.shape)
    # Compute and print R^2 and RMSE
#    print("R^2 Score /Accuracy for train data: {}".format(reg.score(X_train_transform, y_train)))
#    print("R^2 Score /Accuracy for test data: {}".format(reg.score(X_test_transform, y_test)))
    rmse_test = np.sqrt(mean_squared_error(y_test,y_pred_on_test))
    rmse_train = np.sqrt(mean_squared_error(y_train,y_pred_on_train))
    print("Root Mean Squared Error on testing data: {}".format(rmse_test))
    print("Root Mean Squared Error on training data: {}".format(rmse_train))
    print()
    #    print('Scatter plot of actual quality vs predicted quality on the test data')
    #    plt.scatter(y_test,y_pred)
    #    plt.xlabel('Actual Quality / y_test  =>')
    #    plt.ylabel('Predicted Quality / y_pred  =>')
    #    plt.show()

    return  [rmse_test,rmse_train, y_pred_on_train]

def plotoflinregwithtwomostcorrelatedfactors():
    arr = []
    
    X_train1, X_test1, y_train1, y_test1 = traintestsplit(df)
    for i in range(X_train1.shape[1]):
        c = X_train1[:, i]
        arr.append(abs(np.corrcoef(c, y_train1)[0][1]))
    
    t1, t2 = sorted(arr)[:2]
    t1_i = arr.index(t1)
    t2_i = arr.index(t2)
    t1_n = df.columns[:-1][t1_i]
    t2_n = df.columns[:-1][t2_i]
    
    
            
    df2 = pd.DataFrame({t1_n : df[t1_n], t2_n : df[t2_n], 'CO' : df.iloc[:,-1]})
    
    #def multi_reg(df):
    X_train, X_test, y_train, y_test = traintestsplit(df2)
    
    x, y= X_train[:,0], X_train[:,1]
    mx=x.mean()
    sx=x.std()
    my=y.mean()
    sy=y.std()
    xc = np.linspace(mx-2*sx,mx+2*sx, 10)
    yc = np.linspace(my-2*sy,my+2*sy, 10)
    X,Y = np.meshgrid(xc, yc)
    Xi=np.array([X.ravel().tolist(),Y.ravel().tolist()])
    reg = LinearRegression()# Create the regressor: reg
    reg.fit(X_train,y_train)
    Z=reg.predict(Xi.T)
    Z=Z.reshape(X.shape)
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    y_pred=reg.predict(X_test)
    ax.scatter(X_train[:100,0],X_train[:100,1],y_train[:100])
    ax.plot_surface(X, Y, Z,color='red',alpha=0.6)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    print("R^2 Score /Accuracy for train data: {}".format(reg.score(X_train, y_train)))
    print("R^2 Score /Accuracy for test data: {}".format(reg.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    print("Root Mean Squared Error: {}".format(rmse))
    plt.show()
    plt.scatter(y_test,y_pred)
    plt.show()
    
plotoflinregwithtwomostcorrelatedfactors()

def plotofpolyregwithtwomostcorrelatedfactors():
    arr = []
    
    X_train1, X_test1, y_train1, y_test1 = traintestsplit(df)
    for i in range(X_train1.shape[1]):
        c = X_train1[:, i]
        arr.append(abs(np.corrcoef(c, y_train1)[0][1]))
    
    t1, t2 = sorted(arr)[:2]
    t1_i = arr.index(t1)
    t2_i = arr.index(t2)
    t1_n = df.columns[:-1][t1_i]
    t2_n = df.columns[:-1][t2_i]
    
    
            
    df2 = pd.DataFrame({t1_n : df[t1_n], t2_n : df[t2_n], 'CO' : df.iloc[:,-1]})
    
    #def multi_reg(df):
    X_train, X_test, y_train, y_test = traintestsplit(df2)
    
    x, y= X_train[:,0], X_train[:,1]
    mx=x.mean()
    sx=x.std()
    my=y.mean()
    sy=y.std()
    xc = np.linspace(mx-3*sx,mx+3*sx, 10)
    yc = np.linspace(my-3*sy,my+3*sy, 10)
    #xc = np.linspace(min(x), max(x), 100)
    #yc = np.linspace(min(y), max(y), 100)
    
    X,Y = np.meshgrid(xc, yc)
    Xi=np.array([X.ravel().tolist(),Y.ravel().tolist()])
    polynomial_features= PolynomialFeatures(degree = 3)
    x_poly = polynomial_features.fit_transform(X_train)
    reg = LinearRegression()# Create the regressor: reg
    Xi=polynomial_features.fit_transform(Xi.T)
    reg.fit(x_poly,y_train)
    Z=reg.predict(Xi)
    Z=Z.reshape(X.shape)
    
    #y_pred = reg.predict(x_poly)
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection = '3d')
    
    ax.scatter(X_train[:,0],X_train[:,1],y_train)
    ax.plot_surface(X, Y, Z, color = 'red',alpha=0.5)
    ax.set_xlabel(t1_n)
    ax.set_ylabel(t2_n)
    ax.set_zlabel('Z Label')
    
plotofpolyregwithtwomostcorrelatedfactors()