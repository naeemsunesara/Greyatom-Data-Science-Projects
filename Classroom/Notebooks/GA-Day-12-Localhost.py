#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:50:07 2019

@author: naeemsunesara
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.datasets import load_boston

'''boston_data = load_boston()

features = boston_data.feature_names
df = pd.DataFrame(boston_data.data, columns= features)

df['MEDV'] = boston_data.target

corr = df.corr()
plt.figure(figsize = (16,10))


#sns.heatmap(corr, annot=True)

X = df.drop(['MEDV'], 1)
y = df['MEDV']
'''
# Feature Elinibation using model stats
'''
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())
'''

# Getting Linear Coefficients 
'''lin_reg = LinearRegression()
model_1 = lin_reg.fit(X, y)
print(model_1.coef_)
'''

# Making Pipeline

# Recursive Feature Elimination
'''
from sklearn.feature_selection import RFE
lin_reg = LinearRegression()
rfe = RFE(lin_reg,4)
rfe.fit(X,y)
print(rfe.ranking_)
a = rfe.support_
#X.columns[a]
print(a)'''



#gapminder

df = pd.read_csv('gapminder.csv')
X = df[['fertility']]
#X = df.drop(['Region','child_mortality'],1)
y = df['child_mortality']

knn = KNeighborsRegressor(n_neighbors = 3, metric='euclidean')
X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.3, random_state = 42)
knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
score = r2_score(y_test,y_pred)
print(score)

'''lin_reg = LinearRegression()
X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.3, random_state = 42)
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
score = r2_score(y_test,y_pred)
print(y_pred)
print(y_test)'''

df1 = pd.DataFrame()
df1['ypred']= y_pred
df1['ytest'] = y_test
print(score)

cor = df.corr()


