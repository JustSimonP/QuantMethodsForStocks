# Data Preprocessing Tools

# Importing the libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
'''
To import data from another folder:
import pandas as pd
pd.read_csv("../data_folder/data.csv")
'''

'''
Matrix of features
We getting rid of last column as it will be our y(prediction)
iloc function is for locate indexes
:(colon) in python means range, without any number it means we take all rows
'''
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# where to look for missing data from 1st to 3rd column
imputer.fit(X[:, 1:3])
# Matrix is updated with mean values
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Encoding categorical data
# Encoding the Independent Variable
# Making columns with values of String(Country: Germany,Poland) to binary vectors(completely new columns)
'''
For example France will have vector 1:0:0, Germany 0:1:0, Other 0:0:1
'''
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# Encoding the Dependent Variable(YES and NO value to 0 and 1)
le = LabelEncoder()
y = le.fit_transform(y)
print(y)


# Always apply feature scalling after splitting the dataset
# Splitting the dataset into the Training set and Test set usig scikit learn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
sc = StandardScaler()
# Pomijamy kolumny z państwami
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)
