#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:35:15 2019

@author: naeemsunesara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')

## Bootstrap Aggregation


X = df.drop(['diabetes'],1)
y = df['diabetes']

cols = list(X.columns)

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(X),columns = cols)

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X,y , random_state = 42, test_size = 0.3)



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression


model1 = DecisionTreeClassifier(random_state = 42)
model2 = KNeighborsClassifier()
model3 = LogisticRegression(random_state = 42)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)


pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

#Voting Classifier Manually

import statistics as sp

final = []
for x in range(len(pred1)):
    final.append(sp.mode([pred1[x],pred2[x], pred3[x]]))
    
final = np.array(final)

from sklearn.metrics import accuracy_score
    
print(accuracy_score(y_test, final))
print(accuracy_score(y_test, pred1))
print(accuracy_score(y_test, pred2))
print(accuracy_score(y_test, pred3))


#Using voting_classifier

from sklearn.ensemble import VotingClassifier

#voting = hard
model = VotingClassifier(estimators=[('dtc', model1), ('knn', model2), ('log', model3)], voting='hard')
model.fit(X_train,y_train)
print(model.score(X_test,y_test))


## Voting Classifier gives same accuracy score as seen above

# voting = soft
model = VotingClassifier(estimators=[('dtc', model1), ('knn', model2), ('log', model3)], voting='soft')
model.fit(X_train,y_train)
print(model.score(X_test,y_test))


from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier(n_estimators = 100, base_estimator = model2, n_jobs=-1, oob_score = True)
bc.fit(X_train, y_train)
print(bc.score(X_test, y_test))

print(bc.oob_score_)



bc = BaggingClassifier(n_estimators = 100, base_estimator = model1, n_jobs=-1, oob_score = True)
bc.fit(X_train, y_train)
print(bc.score(X_test, y_test))

print(bc.oob_score_)



bc = BaggingClassifier(n_estimators = 100, base_estimator = model3, n_jobs=-1, oob_score = True)
bc.fit(X_train, y_train)
print(bc.score(X_test, y_test))

print(bc.oob_score_)

#Using Random Forest 

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=80,min_samples_split=0.19, random_state=42)

rfc.fit(X_train,y_train)

rfc.score(X_test,y_test)

ok = pd.Series(rfc.feature_importances_, index= list(X))

ok = ok.sort_values(ascending=True)

ok.plot(kind = "barh")




####### Completed with Diabetes Problem #######


### Loan Prediction Problem Started ###

df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

df = df.drop(['Loan_ID'], 1)

miss = df.isna().sum()[df.isna().sum()>0].index.tolist()

for x in miss:
    df[x].fillna(df[x].mode()[0], inplace = True)

    
categorical = df.select_dtypes(exclude=np.number)
    
from sklearn.preprocessing import LabelEncoder    

numerical = df.select_dtypes(include = np.number)    
    
le = LabelEncoder()    

for x in list(categorical):
    categorical[x] = le.fit_transform(categorical[x])    
    

df = pd.concat([numerical,categorical],1)    
    
X = df.drop(['Loan_Status','Married','Dependents','Education','Self_Employed','Gender'],1)  
y = df['Loan_Status']

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, min_samples_split=0.12, random_state=42)

X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.3, random_state=42)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print(accuracy_score(y_test,y_pred))

ok = pd.Series(rfc.feature_importances_, index= list(X))

ok = ok.sort_values(ascending=True)

ok.plot(kind = "barh")



















  