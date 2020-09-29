#B18030-Tushar Goyal
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

global_accuracy = []

def load_dataset(path_to_file):
    df = pd.read_csv(path_to_file)
    return df

def normalization(df):
    feature_df = df.drop('class', axis=1)
    steps=[('scaler',MinMaxScaler()),('pca',PCA(n_components=4))]
    scalernfit=Pipeline(steps)
    df_norm = pd.DataFrame(scalernfit.fit_transform(feature_df))
    df_norm['class'] = df['class']
    return df_norm


def standardize(df):
    feature_df = df.drop('class', axis=1)
    steps=[('scaler',StandardScaler()),('pca',PCA(n_components=2))]
    scalernfit=Pipeline(steps)
    df_standard = pd.DataFrame(scalernfit.fit_transform(feature_df))
    df_standard['class'] = df['class']
    return df_standard


def traintestsplit(df):
    X = df.iloc[:, 0:-1].values
    y = df['class'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,stratify=y, random_state=42,shuffle=True)
    return X_train, X_test, y_train, y_test


def knnclassification(df):
    X_train, X_test, y_train, y_test = traintestsplit(df)
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
    
def naiveclassification(df):
    X_train, X_test, y_train, y_test = traintestsplit(df)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Accuracy for Naive Bayes Classifier: ",gnb.score(X_test,y_test))
    print("Confusion matrix  =\n", confusion_matrix(y_test, y_pred))
    print('\n')
   
def prob(x, w, mean, cov):
    p = 0
    for i in range(len(w)):
        p += w[i] * scipy.stats.multivariate_normal.pdf(x, mean[i], cov[i], allow_singular=True)
        
    return p

#B18030-Tushar Goyal
def gmmbayes(df,k):
    
    df_0=df[df['class']==0]
    df_1=df[df['class']==1]
    
    prior0=len(df_0)/(len(df_0)+len(df_1))
    prior1=1-prior0
    X_train0, X_test0, y_train0, y_test0 = traintestsplit(df_0)
    X_train1, X_test1, y_train1, y_test1 = traintestsplit(df_1)
    
    test = np.concatenate((X_test0, X_test1))
    pred = np.concatenate((y_test0, y_test1))
    
    gmm = GaussianMixture(n_components=k)
    gmm.fit(X_train0)    

#After our model has converged, the weights, means, and covariances should be solved! We can print them out.

#    print("gmm mean_ ", gmm.means_)    
    gmm2 = GaussianMixture(n_components=4)
    gmm2.fit(X_train1)
    
    ypred = []    
    for x in test:
        ypred.append(  0 if prob(x, gmm.weights_, gmm.means_, gmm.covariances_)*prior0\
                           > prob(x, gmm2.weights_, gmm2.means_, gmm2.covariances_)*prior1 else 1 )
    print("Accuracy for GMM  Bayes Classifier with ",k,"components")
    print(accuracy_score(pred, ypred))
    print(confusion_matrix(pred, ypred))
    
    return accuracy_score(pred, ypred)
#---------------------------------------------------------------------------------------------------------------------
#Assignment-7 
    
def main():

    df = load_dataset('pima-indians-diabetes.csv')
    df_norm = normalization(df)
    df_standard = standardize(df)

    print("Original data\n\n")
    knnclassification(df)
    print("\n\n************************************************************\n\n")


    print("Normalized data\n\n")
    knnclassification(df_norm)
    print("\n\n*************************************************************\n\n")


    print("Standardise data\n\n")
    knnclassification(df_standard)

    print("\n\n***********Combined plot for Normalised and Standardised*********\n\n")

    x = list(range(1, 25, 2))
    plt.plot(x, global_accuracy[0], label = "PCA on Original data")
    plt.plot(x, global_accuracy[1], label = "PCA on Normalized data")
    plt.plot(x, global_accuracy[2], label = "PCA on Standardized data")
    plt.legend()
    plt.xlabel("Number of neighbors",fontweight='bold')
    plt.ylabel("Accuracy")
    plt.savefig('Accuracy_vs_k_with_PCA_on_original_normalised_standardised.png')
    plt.show()

    print("\n\n*******Naive Bayes Confusion Matrix and its accuracy score*********\n\n")
    naiveclassification(df_standard)
    
    print("\n\n******* GMM Naive Bayes Confusion Matrix and its accuracy score*********\n\n")
    
    l=[1,2,4,8,12,16]
    acc=[]
    for i in l:
        acc.append(gmmbayes(df,i))
    plt.plot(l,acc)
    plt.ylabel("Accuracy ")
#    plt.x
    
    print("The distribution of the given data is Bimodal")
    
main()
