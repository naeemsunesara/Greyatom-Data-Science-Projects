#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:55:41 2019

@author: naeemsunesara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')

X = df.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))

plt.figure(figsize=(24,12))
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_ac = ac.fit_predict(X)

plt.figure(figsize=(16,8))
plt.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_ac == 2, 0], X[y_ac == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_ac == 3, 0], X[y_ac == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_ac == 4, 0], X[y_ac == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

### Adding Age Feature

X = df.iloc[:,[2,3,4]].values

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))

plt.figure(figsize=(24,12))
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')

y_ac = ac.fit_predict(X)

### here convert it to Dataframe and mask the Cluster number to get there features and get insights


### dendrogram is to get the number of clusters 


## airbnb problem

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

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for x in list(categorical):
    categorical[x] = le.fit_transform(categorical[x])


df = pd.concat([numerical,categorical], 1)


X = df.iloc[:,[4,5,7]].values

plt.figure(figsize=(24,12))
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()

ac = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

y_ac = ac.fit_predict(X)



from sklearn.cluster import KMeans













