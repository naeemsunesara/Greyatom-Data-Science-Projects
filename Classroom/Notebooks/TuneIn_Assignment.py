#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:22:11 2019

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

import plotly.plotly as py


df = pd.read_csv('sf_business_dataset.csv', encoding='LATIN-1')

cols = df.columns
print(cols)


### 1st Problem
df['City'] = df['City'].str.replace('+',' ').str.lower()
sf = df[(df['City'] == 'san francisco') & (df['Business End Date'].isna())]
sf['Source Zipcode'] = sf['Source Zipcode'].fillna(sf['Source Zipcode'].mode()[0])

print(sf['Source Zipcode'].value_counts())

### 2nd Problem




