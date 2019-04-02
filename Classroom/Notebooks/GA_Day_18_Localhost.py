#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:11:15 2019

@author: naeemsunesara
"""

import pandas as pd
import numpy as np


####### Heart Problem ######

df = pd.read_csv('heart.csv')

df = df.dropna()

df = df.drop(['Unnamed: 0'],1)

numerical = df.select_dtypes(include=np.number)
nonnumerical = df.select_dtypes(exclude = np.number)

nonnumerical['Ca'] = numerical['Ca']

numerical = numerical.drop(['Ca'],1)


num_cols = list(numerical.columns)
nonnum_cols = list(nonnumerical.columns)


from sklearn.preprocessing import MinMaxScaler, LabelEncoder

scaler = MinMaxScaler()

numerical = scaler.fit_transform(numerical)
numerical = pd.DataFrame(numerical,columns=num_cols)

le = LabelEncoder()

for x in nonnum_cols[:-1]:
    nonnumerical[x] =le.fit_transform(nonnumerical[x])
    

df = pd.concat([numerical,nonnumerical],1)
df = df.dropna()


X = df.drop(['AHD'],1)
y = df['AHD']


from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, cross_val_score

import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts


X_train, X_test, y_train,y_test = tts(X,y,test_size = 0.3, random_state = 42)

log_reg = LogisticRegression(C = 0.3)

log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_pred)


X = df.drop(['AHD'],1)
y = df['AHD']

X = sm.add_constant(X)
model = sm.Logit(y,X).fit()
print(model.summary())



from sklearn.feature_selection import RFE

rfe = RFE(log_reg, 5)
rfe.fit(X,y)
a = rfe.support_
print(rfe.ranking_)
print(X.columns[a])


from sklearn.tree import DecisionTreeClassifier


dtc = DecisionTreeClassifier()

params = {'max_depth':np.arange(1,10),
          'min_samples_split':np.arange(2,10)}


dtc_cv = GridSearchCV(dtc, param_grid=params,cv=5)

dtc_cv.fit(X,y)
print(dtc_cv.best_estimator_)

X = df[['Ca','ChestPain','Thal','Oldpeak','MaxHR']]
y = df['AHD']

dtc = DecisionTreeClassifier(max_depth = 3, min_samples_split = 2)

X_train, X_test, y_train,y_test = tts(X,y,test_size = 0.3, random_state = 42)

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

imp_features = pd.Series(dtc.feature_importances_, index = X.columns)

from sklearn.metrics import confusion_matrix, classification_report         

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



cross_val_score(dtc, X, y, cv=10, scoring='roc_auc')



###### Heart Problem Completed #####


##### Logistic Regression Boundary #####


### Code Given by Arunabh ###

### College Problem ####

df = pd.read_csv('bank.csv', sep = ';')


df.drop()




































