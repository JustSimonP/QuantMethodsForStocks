# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

bostonHousingDataset = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
print(bostonHousingDataset.head())



class OrdinaryLeastSquares(object):
    def __init__(self):
        self.coefficients = []
        
    def _reshape_x(self, X):
        return X.reshape(-1,1)

#generate the vector and concatenate it
    def _concateate_ones(self, X):
        ones = np.ones(shape=X.shape[0]).reshape(-1,1)
        return np.concatenate((ones,X),1)
    
    def fit(self, X, y):
        if len(X.shape) == 1: X= self._reshape_x(X)
        X = self._concateate_ones(X)
        self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        
    def predict(self,entry):
        b0 = self.coefficients[0]
        other_betas = self.coefficients[1:]
        prediction = b0
        for xi, bi in zip(entry, other_betas): prediction += (bi * xi)
        return prediction
    
    
    
    
    
    
X = bostonHousingDataset.drop('medv', axis=1).values
y = bostonHousingDataset['medv'].values
model  = OrdinaryLeastSquares()
model.fit(X,y)
print(model.coefficients)

#Prediction for first row of X
model.predict(X[0])

#Prediction for all rows
y_preds = []

for row in X: y_preds.append(model.predict(row))

pd.DataFrame({
    'Actual': y,
    'Predicted': np.ravel(y_preds)
    })
      
