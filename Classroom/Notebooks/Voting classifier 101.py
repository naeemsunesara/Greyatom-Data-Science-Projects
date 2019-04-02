#!/usr/bin/env python
# coding: utf-8

# In[19]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
df = pd.read_csv("diabetes.csv")
X = df.drop(["diabetes"],1)
y = df["diabetes"]
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
cols = list(X)
X = pd.DataFrame(scaler.fit_transform(X),columns=cols)
from sklearn.model_selection import train_test_split as tts
X_train, X_test,y_train,y_test= tts(X,y,random_state = 42, test_size = 0.3)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
model1 = DecisionTreeClassifier(random_state=42)
model2 = KNeighborsClassifier()
model3= LogisticRegression(random_state=42)
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
pred1=model1.predict(X_test)
pred2=model2.predict(X_test)
pred3=model3.predict(X_test)


# In[2]:


# import statistics as sp
# final_pred = np.array([])
# for i in range(0,len(x_test)):
#     final_pred = 
#     np.append(final_pred, sp.mode([pred1[i], pred2[i], pred3[i]]))


# In[33]:


final=[]
import statistics as sp
for x in range(len(pred1)):
    final.append((sp.mode([pred1[x], pred2[x],pred3[x]])))
    


# In[36]:


import numpy as np
final = np.array(final)


# In[38]:


from sklearn.metrics import accuracy_score


# In[42]:


accuracy_score(y_test,pred3)


# In[44]:


from sklearn.ensemble import VotingClassifier

model = VotingClassifier(estimators=[('dtc', model1), ('knn', model2), ('log', model3)], voting='hard')
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[13]:





# In[14]:





# In[15]:





# In[17]:





# In[ ]:




