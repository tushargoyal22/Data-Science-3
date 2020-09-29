import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
import statsmodels.api as sm

def readfile(filename): return pd.read_csv(filename,header=0, index_col=0, parse_dates=True, squeeze=True)
    
def seriesplot(series,X,Y,Title): plt.plot([i for i in range(141)],series.values);plt.xlabel(X);plt.ylabel(Y);plt.title(Title);plt.show()

def PrintCorrMatrix(series): print('\nCorrelation Matrix:\n',pd.concat([series.shift(1), series], axis=1).corr(method ='pearson'))

def autocorrplot(series,n_lags): sm.graphics.tsa.plot_acf(series, lags=n_lags);plt.show()

def persistence_model(df,testsize,i): return df.iloc[(-1*testsize)+i-1]

def modeldetails(model):
    print('\nBest Lagging Period: %s' % model.k_ar)
    print('\nCoefficients:')
    print("c00: "+' '*int(model.params[0]>0)+'%.6f' %model.params[0])
    for i in range(1,len(model.params)):print('c'+str(i).zfill(2)+': '+' '*int(model.params[i]>0)+'%.6f' %model.params[i])

def predictionsanalysis(test,predictions):
    print("\nPrediction Analysis:")
    #for i in range(len(predictions)):print('expected: %f | predicted: %f' % (test[i],predictions[i]))
    print(36*'-'+'\nTest MSE: %f' % mean_squared_error(test, predictions)**(0.5))
    plt.plot(test.values);plt.plot(predictions, color='red');plt.show()

def main():
    filename = "SoilForce.csv"
    #filename = "daily-min-temperatures.csv"
    #filename = "Rain.csv"
    
    #Read File
    series = readfile(filename)
    
    #Attribute Name
    subject = series.name
    
    #Plot the data vs time
    seriesplot(series,"Date",subject,"Date v/s "+subject)
    
    #Correlation Matrix
    PrintCorrMatrix(series)
    
    #AutocorrelationPlot for different values of lag period
    autocorrplot(series,35)
    
    #Test Size
    #testsize = 7
    testsize = len(list(series))//2
   
    #Splitting the data
    train, test = series.iloc[1:(-1*testsize)], series.iloc[(-1*testsize):]
    
    #Persistence Model Predictions
    predictionsA = [persistence_model(series,testsize,i) for i in range(testsize)]
    predictionsanalysis(test,predictionsA)
    
    #AutoRegression Model Predictions
    model = AR(train.values).fit()
    modeldetails(model)
    predictionsB = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    predictionsanalysis(test,predictionsB)
    
if __name__ == "__main__":
    main()