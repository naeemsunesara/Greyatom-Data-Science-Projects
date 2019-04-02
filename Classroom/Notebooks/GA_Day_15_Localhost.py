#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 09:58:48 2019

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

from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('Credit.csv')


X = df.drop(['Unnamed: 0','Balance'],1)
X = pd.get_dummies(X)
cols = list(X.columns)

y = df['Balance']

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=cols)

cols_cat = ['Gender_ Male','Student_No','Married_Yes','Ethnicity_Asian','Ethnicity_Caucasian']

cols_final = cols[:6] + cols_cat

X = X[cols_final]

for x in cols_cat:
    X[x] = X[x].astype('category')


#df = pd.concat([num,cat],axis = 1)

plt.figure(figsize=(15,8))


df = pd.concat([X,pd.DataFrame(y)],axis = 1)

num = df.select_dtypes(include=[np.number])

cat = df.select_dtypes(include=['category'])

df = df.astype({"Gender_ Male": float, "Student_No": float, "Married_Yes":float, "Ethnicity_Asian":float,"Ethnicity_Caucasian":float})


X = df.drop(['Balance','Married_Yes','Ethnicity_Asian','Ethnicity_Caucasian','Education','Gender_ Male','Age','Cards'], 1)
y = df['Balance']

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())





X = df.drop(['Balance'], 1)
y = df['Balance']

from sklearn.feature_selection import RFE
lin_reg = LinearRegression()

rfe = RFE(lin_reg,4)
rfe.fit(X,y)
print(rfe.ranking_)
a = rfe.support_
X.columns[a]
print(a) 


import statsmodels.formula.api as smf
results = smf.ols('Balance~Limit+Student_No', data=df).fit()
print(results.summary())


from sklearn.linear_model import RidgeCV, LassoCV
reg = RidgeCV()
model = reg.fit(X,y)
print(model.alpha_)
print(reg.score(X,y))


las = LassoCV()
model = las.fit(X,y)
print(model.alpha_)
print(las.score(X,y))

coef =  pd.Series(reg.coef_, index=(X.columns))
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")    



#####  Gapminder ####

df = pd.read_csv('gapminder.csv')

X = df.drop(['child_mortality'], 1)
y = df['child_mortality']

X = pd.get_dummies(X)

cols = list(X.columns)

las = LassoCV()
model = las.fit(X,y)
print(model.alpha_)
print(las.score(X,y))

coef =  pd.Series(las.coef_, index=(X.columns))
print(coef)




#### Credit card Fraud

import scipy.stats as st

df = pd.read_csv('creditcard.csv')

zero = df[df['Class'] == 0]

one = df[df['Class'] == 1]

zero = zero.sample(1492, random_state=42)
df = pd.concat([one,zero])

X = df.drop(['Class'],1)
y = df['Class']

cols = list(X.columns)
while len(cols)>0:
   pvalue = []
   temp = X[cols]
   X_const = sm.add_constant(temp)
   model = sm.Logit(y,X_const).fit()
   pvalue = pd.Series(model.pvalues.values,index=X_const.columns)
   maxpindex = pvalue.idxmax()
   if(pvalue[maxpindex]>0.05):
       cols.remove(maxpindex)
   else:
       break;

print(cols)

X = X[cols]

logistic = LogisticRegression()
score = cross_val_score(logistic,X,y,cv=20)
print(score)
print("Worst Model: {}".format(score.min()))
print("Mean Model: {}".format(score.mean()))