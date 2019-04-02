#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:27:35 2019

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


# Logistic Regression

df = pd.read_excel('beer_cleaned.xlsx')

X = df.drop(['Weekend'],1)
y = df['Weekend']

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

X_train,X_test, y_train,y_test = tts(X,y,test_size = 0.3, random_state = 42)

log_reg.fit(X_train,y_train)

y_pred = log_reg.predict(X_test)

print(accuracy_score(y_test,y_pred))


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))


### Logistic Heart Disease Problem

df = pd.read_csv('framingham.csv')

df = df.dropna()

# To remove Class Imbalance and Increase recall

zeros = df[df['TenYearCHD'] == 0][:700]
ones = df[df['TenYearCHD'] == 1]

df_new = pd.concat([zeros,ones], axis = 0)

X = df_new.drop(['TenYearCHD'],1)
y = df_new['TenYearCHD']


log_reg = LogisticRegression()

X_train,X_test, y_train,y_test = tts(X,y,test_size = 0.3, random_state = 42)

log_reg.fit(X_train,y_train)

y_pred = log_reg.predict(X_test)

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))


##

import scipy.stats as st



X = df_new.drop(['TenYearCHD','education','diabetes','currentSmoker','diaBP','heartRate','BMI','BPMeds','prevalentStroke','totChol'],1)
y = df_new['TenYearCHD']


X = sm.add_constant(X)
st.chisqprob = lambda chisq, df_new: st.chi2.sf(chisq,df_new)
model = sm.Logit(y,X)
result = model.fit()
print(result.summary())



#### Using KNN Classifier , SMOTE , Gridsearch CV
df = pd.read_csv('framingham.csv')
df = df.dropna()
zeros = df[df['TenYearCHD'] == 0]
ones = df[df['TenYearCHD'] == 1]
df_new = pd.concat([zeros,ones], axis = 0)




X = df_new[['age','glucose','male','sysBP','totChol','cigsPerDay']]
y = df_new['TenYearCHD']




from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X, y.ravel())

X_new = pd.DataFrame(X_res, columns=list(X.columns))
y_new = pd.Series(y_res)









