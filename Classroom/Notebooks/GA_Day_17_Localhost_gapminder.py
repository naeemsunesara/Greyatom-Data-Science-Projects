#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:19:05 2019

@author: naeemsunesara
"""

import pandas as pd 
import numpy as np
df = pd.read_csv('gapminder.csv') 

from sklearn.model_selection import train_test_split as tts, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import *

X = df.drop(['child_mortality'],1)
y = df['child_mortality']

X = pd.get_dummies(X)

dtr = DecisionTreeRegressor()
X_train, X_test, y_train , y_test = tts(X,y,test_size = 0.3, random_state = 42)

dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))



#### Decision Tree Using Grid Search CV

params = {'max_depth':np.arange(1,10),
          'min_samples_split':np.arange(2,10),
          'max_features':np.arange(1,10)}



dtr_cv = GridSearchCV(dtr, param_grid = params, cv=5)

dtr_cv.fit(X,y)

print(dtr_cv.best_params_)
print(dtr_cv.best_score_)


### KNeigbors Reqressor

knn = KNeighborsRegressor()

params  = {'metric':['minkowski','euclidean','manhattan'],
           'n_neighbors':np.arange(1,10)
        }

knn_cv = GridSearchCV(knn, param_grid = params, cv = 5)

knn_cv.fit(X,y)

print(knn_cv.best_estimator_)
print(knn_cv.best_score_)































