#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd


# In[69]:


df = pd.read_csv("H-1B_Disclosure_Data_FY17.csv")


# In[70]:


df.shape


# In[71]:


# d = df.isnull().sum()[(df.isnull().sum()/df.shape[0])*100>30]


# In[72]:


# drop = d.index.tolist()


# In[73]:


# df = df.drop(drop,1)


# In[ ]:





# In[ ]:





# In[74]:


df.head()


# In[75]:


cols = list(df)
drop = pd.DataFrame(df.isnull().sum()[((df.isnull().sum()/df.shape[0])*100)>30]).index.tolist()


# In[76]:


df = df.drop(drop,1)


# In[10]:


# !pip install fancyimpute
# from fancyimpute import KNN    
# # X is the complete data matrix
# # X_incomplete has the same values as X except a subset have been replace with NaN

# # Use 3 nearest rows which have a feature to fill in each row's missing features
# # X_filled_knn = KNN(k=3).complete(X_incomplete)


# In[77]:


df["EMPLOYMENT_START_DATE"].isnull().sum()


# In[12]:


# def impute_missing(column_name):
#     if int(df[column_name].isnull().sum())>0:
#         df[column_name]=df[column_name].fillna(df[column_name].mean())
#     return df[column_name]


# In[13]:


type(int(df["EMPLOYER_ADDRESS"].isnull().sum()))


# In[78]:


cols = list(df)


# In[79]:


df[cols]=df[cols].fillna(df.mode().iloc[0])


# In[80]:


df.shape


# In[81]:


df.isnull().sum()


# In[82]:


drop= list(df)[:2]


# In[83]:


df= df.drop(drop,1)


# In[84]:



print (len(df["WORKSITE_STATE"].unique()))


# In[85]:


df = df.drop(["WORKSITE_CITY", "WORKSITE_COUNTY", "WORKSITE_POSTAL_CODE"],1)


# In[86]:


list(df)


# In[87]:


df = df.drop(["EMPLOYER_NAME","EMPLOYER_ADDRESS","EMPLOYER_CITY",
"EMPLOYER_POSTAL_CODE","EMPLOYER_COUNTRY","EMPLOYER_PHONE", "AGENT_ATTORNEY_NAME"],1)


# In[88]:


df.shape


# In[89]:


df = df.drop(["PW_SOURCE_OTHER", "SOC_NAME"],1)


# In[90]:


y = df["CASE_STATUS"]


# In[91]:


df["CASE_STATUS"].value_counts()


# In[92]:


df_1 = df[df["CASE_STATUS"]=="DENIED"]


# In[93]:


df_2 = df[df["CASE_STATUS"]=="WITHDRAWN"]
df_3 = df[df["CASE_STATUS"]=="CERTIFIED-WITHDRAWN"]
df_4 = df[df["CASE_STATUS"]=="CERTIFIED"]


# In[94]:


df_2 = df_2.sample(n = 8480)


# In[95]:


df_3 = df_3.sample(n = 8480)
df_4 = df_4.sample(n = 8480)


# In[99]:


df = pd.concat([df_1,df_2,df_3,df_4])


# In[100]:


df.shape


# In[101]:


df["CASE_STATUS"].value_counts()


# In[104]:


import numpy as np
numerical = df.select_dtypes(include = np.number)
categorical = df.select_dtypes(exclude = np.number)


# In[109]:


categorical["CASE_SUBMITTED"]=  pd.to_datetime(categorical["CASE_SUBMITTED"])
categorical["DECISION_DATE"]=  pd.to_datetime(categorical["DECISION_DATE"])
categorical["EMPLOYMENT_START_DATE"]=  pd.to_datetime(categorical["EMPLOYMENT_START_DATE"])
categorical["EMPLOYMENT_END_DATE"]=  pd.to_datetime(categorical["EMPLOYMENT_END_DATE"])


# In[112]:


categorical["decision_period"]= (categorical["DECISION_DATE"] - categorical["CASE_SUBMITTED"]).dt.days


# In[114]:


categorical["employment_period"] = (categorical["EMPLOYMENT_END_DATE"] - categorical["EMPLOYMENT_START_DATE"]).dt.days


# In[116]:


drop = ["CASE_SUBMITTED", "DECISION_DATE", "EMPLOYMENT_END_DATE", "EMPLOYMENT_START_DATE"]


# In[119]:


categorical= categorical.drop(drop,1)


# In[121]:


df = pd.concat([numerical,categorical],1)


# In[122]:


df.shape


# In[124]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[126]:


date = categorical[["decision_period", "employment_period"]]


# In[129]:


categorical= categorical.drop(["decision_period", "employment_period"],1)


# In[142]:


cols = list(categorical)


# In[151]:


for x in cols:
    categorical[x] = categorical[x].astype("category")


# In[152]:


categorical = categorical.drop(["NAICS_CODE"],1)


# In[153]:


for x in list(categorical):
    categorical[x] = le.fit_transform(categorical[x])


# In[155]:


df = pd.concat([numerical,categorical],1)


# In[158]:


X = df.drop(["CASE_STATUS"],1)
y = df["CASE_STATUS"]


# In[159]:


from sklearn.tree import DecisionTreeClassifier


# In[167]:


dtc = DecisionTreeClassifier(max_features=10, min_samples_split=0.12)


# In[168]:


from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test = tts(X,y,test_size = 0.3,random_state = 42)


# In[169]:


dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)


# In[170]:


from sklearn.metrics import accuracy_score


# In[171]:


accuracy_score(y_test,y_pred)


# In[165]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[166]:


knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:




