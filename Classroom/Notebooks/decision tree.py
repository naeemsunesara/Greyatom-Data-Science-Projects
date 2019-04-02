#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd


# In[99]:


df = pd.read_csv("Heart.csv")


# In[100]:


df.head()


# In[101]:


df.isnull().sum()


# In[102]:


df = df.dropna()


# In[103]:


df.isnull().sum()


# In[104]:


df = df.drop(["Unnamed: 0"],1)


# In[105]:


df.head()


# In[70]:





# In[71]:


df.shape


# In[106]:


df["AHD"].value_counts()


# In[107]:


import numpy as np
numerical = df.select_dtypes(include = np.number)
nonnumerical = df.select_dtypes(exclude = np.number)
nonnumerical["Ca"] = numerical["Ca"]
numerical = numerical.drop(["Ca"],1)


# In[108]:


from sklearn.preprocessing import MinMaxScaler


# In[109]:


cols = list(numerical)


# In[110]:


cols


# In[111]:


scaler = MinMaxScaler()


# In[ ]:





# In[ ]:





# In[112]:


numerical = scaler.fit_transform(numerical)


# In[113]:


numerical = pd.DataFrame(numerical)


# In[114]:


numerical.columns = cols


# In[117]:





# In[118]:


from sklearn.preprocessing import LabelEncoder


# In[119]:


le = LabelEncoder()


# In[120]:


for x in list(nonnumerical.iloc[:,:-1]):
    nonnumerical[x] = le.fit_transform(nonnumerical[x])


# In[124]:


nonnumerical.isnull().sum()


# In[125]:


df = pd.concat([numerical,nonnumerical],1)


# In[129]:


df = df.dropna()


# In[131]:


df.isnull().sum()


# In[132]:


y = df["AHD"]
X = df.drop(["AHD"],1)


# In[133]:


from sklearn.tree import DecisionTreeClassifier


# In[134]:


dtc = DecisionTreeClassifier()
from sklearn.model_selection import train_test_split as tts


# In[135]:


X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.2, random_state = 42)


# In[137]:


dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)


# In[138]:


from sklearn.metrics import accuracy_score


# In[139]:


accuracy_score(y_test,y_pred)


# In[140]:


import os


# In[141]:


os.getcwd()


# In[ ]:




