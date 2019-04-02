#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:36:34 2019

@author: naeemsunesara
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
#### Credit card Fraud

import scipy.stats as st

df = pd.read_csv('breast_cancer.csv')
df = df.drop(['id','Unnamed: 32'],1)
le = preprocessing.LabelEncoder()
le.fit(df['diagnosis'])
df['diagnosis'] = le.transform(df['diagnosis']) 
X = df.drop(['diagnosis'],1)
y = df['diagnosis']
cols = list(X.columns)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X , columns= cols)




from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

knn = KNeighborsClassifier()

params = {"n_neighbors":np.arange(1,20)}
model = GridSearchCV(knn , param_grid = params, cv = 10)
model.fit(X,y)


scoring = ['accuracy', 'precision','roc_auc','recall']
#auc_score = cross_val_score(knn,X,y,cv=10,scoring=scoring)
cv_results = cross_validate(model.best_estimator_, X, y,scoring=scoring, cv=10)


from sklearn.feature_selection import RFE
log_reg = LogisticRegression()

rfe = RFE(log_reg,10)
rfe.fit(X,y)
print(rfe.ranking_)
a = rfe.support_
X.columns[a]
print(a) 


X = sm.add_constant(X)
model = sm.Logit(y,X).fit()
print(model.summary)
