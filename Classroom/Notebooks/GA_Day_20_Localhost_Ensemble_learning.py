#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 10:13:24 2019

@author: naeemsunesara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('hr_employee.csv')

categorical = df.select_dtypes(exclude = np.number )

numerical = df.select_dtypes(include = np.number)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for x in list(categorical):
    categorical[x] = le.fit_transform(categorical[x])
    
    
df = pd.concat([numerical,categorical], 1)    
    

X = df[['MonthlyIncome','YearsAtCompany','Age','StockOptionLevel','OverTime']]
y = df['Attrition']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.3, random_state = 42)

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

## Class Imbalance accuracy score = 0.86167 not good

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

impf = pd.Series(rfc.feature_importances_, index = list(X))

impf = impf.sort_values(ascending=True)

impf.plot(kind = "barh")
# from imblearn.over_sampling import SMOTE



###### Grid Search CV #########

from sklearn.model_selection import GridSearchCV

params = {'max_depth':np.arange(1,10),
          'min_samples_split':np.arange(2,10),
          'max_features':np.arange(2,5)}

model = GridSearchCV(rfc, param_grid=params,cv=5)

model.fit(X,y)

#################   



##### Class Weights Code #####
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(y),
                                                y)
class_weight=dict(enumerate(class_weights))

############


#### ADABOOST ####


from sklearn.ensemble import AdaBoostClassifier


abc = AdaBoostClassifier(rfc)


abc.fit(X_train,y_train)

y_pred = abc.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

## Class Imbalance accuracy score = 0.86167 not good

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

##### airbnb Problem ########


df = pd.read_csv('dc_airbnb.csv')


df['host_response_rate'] = df['host_response_rate'].str.replace('%','')


df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%','')

df['price'] = df['price'].str.replace('$','')

df['cleaning_fee'] = df['cleaning_fee'].str.replace('$','')

df['security_deposit'] = df['security_deposit'].str.replace('$','')

df = df.drop(['cleaning_fee','security_deposit','latitude','longitude'],1)

miss = df.isna().sum()[df.isna().sum()>0].index.tolist()


for x in miss:
    df[x] = df[x].fillna(df[x].mode()[0])

df['price'] = df['price'].str.replace(',','')

df = df.astype({"host_response_rate": int, "host_acceptance_rate": int,"price":float})

df = df.drop(['city','state'],1)

categorical = df.select_dtypes(exclude = np.number)

numerical = df.select_dtypes(include = np.number)

for x in list(categorical):
    categorical[x] = le.fit_transform(categorical[x])



df = pd.concat([numerical,categorical], 1)

df2 = df[df['price']<=200]

X = df2[['bedrooms','bathrooms','maximum_nights','accommodates']]
y = df2['price']


from sklearn.ensemble import RandomForestRegressor


rfr = RandomForestRegressor()

X_train, X_test, y_train, y_test = tts(X,y, test_size=0.3, random_state=42)

rfr.fit(X_train,y_train)

y_pred = rfr.predict(X_test)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))

impf = pd.Series(rfr.feature_importances_, index = list(X))

impf = impf.sort_values(ascending=True)

impf.plot(kind = "barh")



























