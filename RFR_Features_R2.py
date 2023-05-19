# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset, [rows, columns]
dataset = pd.read_csv('48_17.5.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
'''
#oneHot Encoding
#imputer = fills averages
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=s=np.nan, strategy='mean')
#Number of cells
temp = X[:, 13:15]
imputer.fit(X[:, 13:15])
X[:, 13:15] = imputer.transform(X[:, 13:15])

#Feature scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X[:,13:]=ss.fit_transform(X[:,13:])
#X_train[:,13:]=ss.fit_transform(X_train[:,13:])
#_test[:,13:]=ss.transform(X_test[:,13:])

#ENCODING CATEGORICAL DATA
#Encoding independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(sparse=False,), [0,1,2,3,4,5,6,7,8,9,10,11,12])], remainder='passthrough')
X = np.array(ct.fit_transform(X),dtype=str)

#Encoding dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)'''

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#Number of cells
temp = X[:, 13:15]
imputer.fit(X[:, 13:15])
X[:, 13:15] = imputer.transform(X[:, 13:15])
X_float = X[:,13:].astype(float) 
X = X[:,0:13]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
i=0
while i < len(X[0]):
    X[:,i]=le.fit_transform(X[:,i].astype(str))
    i+=1

y = le.fit_transform(y)
X_added = np.concatenate((X, X_float), axis=1) #NEW ARRAY CONCATENATED



#Splitting into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_added, y, test_size = 0.2, random_state = 1)

#Random forest regression code
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100    , random_state = 0)
regressor.fit(X_added, y)
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred) 
print(r2)


print(regressor.feature_importances_)
np_feature_names = ["material", "type", "shape", "coat", "syntehsis_method", "surface charge", "cell type", "cell tissue", "cell morphology", "cell age", "cell line", "test", "test indicator", "concentration", "number of cells", "time"]
plt.barh(np_feature_names, regressor.feature_importances_)

    


