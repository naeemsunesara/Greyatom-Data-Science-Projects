# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

### Linear Regression ###
'''df = pd.read_csv("50_Startups.csv")
df_state = pd.get_dummies(df['State'])
df = pd.concat([df,df_state], axis = 1)
df = df.drop(['State'],1)


X = df.drop(['Profit','Florida'],1)
y = df['Profit']
X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.3, random_state = 42)
lin_reg = LinearRegression()
model_1 = lin_reg.fit(X_train, y_train)
y_pred = model_1.predict(X_test)
print(cross_val_score(lin_reg, X, y, groups=None, scoring='r2', cv=10))
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))
print(lin_reg.coef_)'''



### Model Stats for Feature Elimination ###
'''df = pd.read_csv("50_Startups.csv")
df_state = pd.get_dummies(df['State'])
df = pd.concat([df,df_state], axis = 1)
df = df.drop(['State'],1)

X = df.drop(['Profit','Florida','New York','California','Administration'],1)
y = df['Profit']

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())'''

df = pd.read_csv("gapminder.csv")
df = df.drop(['Region'],1)

X = df.drop(['life','population','fertility','BMI_male','CO2'], 1)
y = df['life']

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())
