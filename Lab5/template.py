###################################################################################

# Import the necessary libraries
"""
Try to use the functionalities of the libraries imported.
For example, rather than converting Pandas dataframe into
a list and then perform calculations, use methods of Pandas
library.
"""
###################################################################################


###################################################################################

# Import or Load the data
def load_dataset(path_to_file):
    """
    Load the dataset using this function and then return the
    dataframe. The function parameters can be as per the code
    and problem. Return the loaded data for preprocessing steps.
    """
###################################################################################


###################################################################################

# Data Preprocessing
"""
- Check for outliers.
- Check for missing values.
- Encoding categorical data
- Standardization/Normalization
- Dimensionality Reduction (PCA)
- Shuffle
- Train/Test Split
"""

def outliers_detection(function_parameters):
    ...
    ...
    ...

def missing_values(function_parameters):
    ...
    ...
    ...
    
def encoding(function_parameters):
    """
    Encode the categorical data in your dataset using One-Hot
    encoding. Very important if your dependent variable is
    categorical.
    """
    
def normalization(function_parameters):
    ...
    ...
    ...
    
def dimensionality_reduction(function_parameters):
    """
    Pass the respective function parameters needed by the function
    and perform dimentionality reduction. Retain the useful and
    significant principal components. Dimensionality reduction
    using PCA comes at a cost of interpretibility. The features in
    original data (age, height, income, etc.) can be intrepreted
    physically but not principal components. So decide accordingly.
    Then return the dimension reduced data.
    """

def shuffle(function_parameters):
    """
    Now your data is preprocessed. Shuffle to 'randomize' the data
    for next step of machine learning. Pass the respective parameters
    needed by the function and shuffle the data. Then return the
    shuffled data for next step of splitting it into training and test
    data.
    """

def train_test_split(function_parameters):
    """
    Now your data is preprocessed and shuffle. It's time to divide it
    into training and test data.
    
    Example:
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
    ...                                 test_size=0.3, random_state=42)

            X: independent features
            Y: dependent features
            test_size: fraction of data to be splitted into test data
            random_state: a seed for random generator to produce same
                            "random" results each time the code is run.

    Now your data is ready for classification.
    """

###################################################################################


###################################################################################

# Perform classification

def classification(function_parameters):
    """
    Pass the respective function parameters and perform classification.
    """

###################################################################################


###################################################################################

# Calculate model evaluation scores like
"""
- F-score
- Confusion Matrix
"""

def f_score(function_parameters):
    ...
    ...
    ...

def confusion_matrix(function_parameters):
    ...
    ...
    ...

###################################################################################