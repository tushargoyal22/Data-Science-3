# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_dataset(path_to_file):
    df = pd.read_csv(path_to_file)
    return df


def traintestsplit(df):
    X = df.iloc[:, 0:-1].values
    y = df['quality'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def traintestsplit_l(df):
    X = df.iloc[:,8].values.reshape(-1,1)
    y = df['quality'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)
    return X_train, X_test, y_train, y_test

#load the dataset
df=load_dataset('winequality-red.csv')
    
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
    plt.xlabel('Ph Value')
    plt.ylabel('Quality')
    plt.legend()
    plt.show()
    
    # Predict on the test data: y_pred
    y_pred = reg.predict(X_test)
    df_p = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df_p.head(10))
    print('-'*60)
    # Compute and print R^2 and RMSE
    print("R^2 Score /Accuracy for train data: {}".format(reg.score(X_train, y_train)))
    print("R^2 Score /Accuracy for test data: {}".format(reg.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    print("Root Mean Squared Error: {}".format(rmse))
    print('-'*60)
    print('Scatter plot of actual quality vs predicted quality on the test data')
    plt.scatter(y_test,y_pred)
    plt.xlabel('y_test/Actual Quality  =>')
    plt.ylabel('y_pred /Predicted Quality =>')
    plt.show()
    print('-'*23+'Q1 Completed'+'-'*24)
#linear_reg()

def multi_reg():
    X_train, X_test, y_train, y_test = traintestsplit(df)    
    reg = LinearRegression()# Create the regressor: reg
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
#    df_p = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#    print(df_p.head(10))
    # Compute and print R^2 and RMSE
    print("R^2 Score /Accuracy for train data: {}".format(reg.score(X_train, y_train)))
    print("R^2 Score /Accuracy for test data: {}".format(reg.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    print("Root Mean Squared Error: {}".format(rmse))
    plt.scatter(y_test,y_pred)
    plt.xlabel('y_test  =>')
    plt.ylabel('y_pred  =>')
    plt.show()
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
    prediction_space_t=polynomial_features.fit_transform(prediction_space)
    y_pred_predictspace=reg.predict(prediction_space_t)#predicted values for training data
    plt.scatter(X_train,y_train)
    # Plot regression line:#a. Plot the best fit line on the training data where x-axis is pH value and y-axis is quality
    plt.plot(prediction_space, y_pred_predictspace, color='black', linewidth=3,label='Curve of degree {}'.format(deg))
    plt.xlabel('Ph Value')
    plt.ylabel('Quality')
    plt.legend()
    plt.show()
    
    return  [rmse_test,rmse_train]

def plots_simple_poly_regression():
    rms_error_test = []
    rms_error_train=[]
    for i in range(2, 8):
        rms_error_test.append(simple_poly_reg(i)[0])
        rms_error_train.append(simple_poly_reg(i)[1])
        
    
    plt.ylim(0.8375,0.8475)
    plt.title('Bar graph of RMSE vs different values of degree of polynomial For Test Data')
    plt.bar(list(range(2,8)),rms_error_test)
    plt.xlabel('Degree of Polynomial p=>')
    plt.ylabel('rmse for model with degree p on test data')
    plt.show()
    
    plt.ylim(0.8550,0.8650)
    plt.title('Bar graph of RMSE vs different values of degree of polynomial For Train Data')
    plt.bar(list(range(2,8)),rms_error_train)
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
    print('(poly deg ',deg,')linear model intercept (b): {:.3f}'.format(reg.intercept_))
#    print(reg.coef_.shape)
    # Compute and print R^2 and RMSE
    print("R^2 Score /Accuracy for train data: {}".format(reg.score(X_train_transform, y_train)))
    print("R^2 Score /Accuracy for test data: {}".format(reg.score(X_test_transform, y_test)))
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
    
    return  [rmse_test,rmse_train]

multi_poly_reg(1)
#multi_poly_reg(2)
#multi_poly_reg(3)
#multi_poly_reg(4)



