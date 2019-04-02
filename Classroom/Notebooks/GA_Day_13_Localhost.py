#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:40:02 2019

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

df = pd.read_excel('output.xlsx')
plt.figure(figsize=(10,8))
plt.scatter(df["X"], df["y"])

lr = LinearRegression()
X = np.array(df['X']).reshape(-1,1)
# reshape to get 1 column in array
y = np.array(df['y'])
lr.fit(X,y)
y_pred = lr.predict(X)
plt.plot(X,y_pred)
print(lr.coef_)

# bringing in outliers

idx = y.argmax()
y[idx] = 200
idx = y.argmin()
y[idx] = -200

plt.scatter(df["X"], y)

lr = LinearRegression()
X = np.array(df['X']).reshape(-1,1)
# reshape to get 1 column in array
y = np.array(y)
lr.fit(X,y)
y_pred = lr.predict(X)
plt.plot(X,y_pred)
print(lr.coef_)

# Ridge Regression

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0.5, normalize = True)
plt.scatter(X,y)
ridge.fit(X,y)
model = ridge.predict(X)
plt.plot(X,model)
print(ridge.coef_)

# Lasso Regression

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.5, normalize = True)
plt.scatter(X,y)
lasso.fit(X,y)
model_l = lasso.predict(X)
plt.plot(X, model_l)
print(lasso.coef_)


# ElasticNet Regression
from sklearn.linear_model import ElasticNet
en = ElasticNet(alpha = 0.5, normalize = True)
plt.scatter(X,y)
en.fit(X,y)
model_en = en.predict(X)
plt.plot(X, model_en)
print(en.coef_)

#### Beer Problem 


df = pd.read_csv('Consumo_cerveja.csv')
df = df.dropna()

df = df.drop(['Data'], 1)


df['temp_max'] = df['Temperatura Maxima (C)'].str.replace(',','.')

df['temp_med'] = df['Temperatura Media (C)'].str.replace(',','.')

df['temp_min'] = df['Temperatura Minima (C)'].str.replace(',','.')

df['Precipitacao (mm)'] = df['Precipitacao (mm)'].str.replace(',','.')

df = df.drop(['Temperatura Maxima (C)','Temperatura Media (C)','Temperatura Minima (C)'], 1)

df= df[['temp_min','temp_med','temp_max','Precipitacao (mm)','Final de Semana','Consumo de cerveja (litros)']]

df = df.astype({"temp_min": float, "temp_med": float, "temp_max":float, "Precipitacao (mm)":float})

corr = df.corr()
plt.figure(figsize = (16,10))
sns.heatmap(corr, annot=True)


X = df.drop(['Consumo de cerveja (litros)'], 1)
y = df['Consumo de cerveja (litros)']

# feature Elimination using Pearson we got

#X = df['temp_max']
#y = df['Consumo de cerveja (litros)']


# Using modelstats

X = df.drop(['Consumo de cerveja (litros)','temp_med','temp_min'], 1)
y = df['Consumo de cerveja (litros)']
 
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())

#adjusted r2 = 0.720


X = df[['temp_max']]
y = df['Consumo de cerveja (litros)']
 
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())


# using only pearson we get r2 = 0.41 which is not correct

X = df[['temp_max','Final de Semana']]
y = df['Consumo de cerveja (litros)']
 
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())

# using only pearson and adding one feature we get r2 = 0.695 as accuracy increased


## Using RFE
X = df.drop(['Consumo de cerveja (litros)'], 1)
y = df['Consumo de cerveja (litros)']

from sklearn.feature_selection import RFE
lin_reg = LinearRegression()
rfe = RFE(lin_reg,3)
rfe.fit(X,y)
print(rfe.ranking_)
a = rfe.support_
X.columns[a]
print(a)

# RFE also giving same features as statsmodel

# so max we got r2 = 0.720


# Now using with Ridge Regression
X = df.drop(['Consumo de cerveja (litros)','temp_med','temp_min'], 1)
y = df['Consumo de cerveja (litros)']


from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0.5, normalize = True)
#plt.scatter(X,y)
ridge.fit(X,y)
model = ridge.predict(X)
#plt.plot(X,model)
print(ridge.coef_)

from sklearn.metrics import r2_score
print(r2_score(y, model)) 


##### CV

from sklearn.model_selection import cross_val_score
X = df.drop(['Consumo de cerveja (litros)','temp_med','temp_min'], 1)
y = df['Consumo de cerveja (litros)']
model = LinearRegression()
cross_val_score(model,X,y, cv= 10)

########

# Ridge CV ----- Elbow method

X = df.drop(['Consumo de cerveja (litros)','temp_med','temp_min'], 1)
y = df['Consumo de cerveja (litros)']

alpha = np.arange(0,2,0.1)
alphas = list(alpha)

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
    
model_ridge = Ridge()
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")    


model = Ridge(alpha=0.25)
model.fit(X,y)
model.score(X,y)


X = df.drop(['Consumo de cerveja (litros)','temp_med','temp_min'], 1)
y = df['Consumo de cerveja (litros)']

alpha = np.arange(0,2,0.1)
alphas = list(alpha)

from sklearn.linear_model import Ridge, RidgeCV,Lasso, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
    
model_lasso = Lasso()
cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")    


model = Lasso(alpha=0.1)
model.fit(X,y)
model.score(X,y)

df = pd.read_csv('cars_2.csv')

