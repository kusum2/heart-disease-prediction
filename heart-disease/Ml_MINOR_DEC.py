#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:


heart = pd.read_csv('heart1.csv')


# In[4]:


heart.isnull().sum()


# In[5]:


heart.head(10)


# In[6]:


heart.info()


# In[7]:


heart.target.value_counts()


# In[8]:


sns.countplot(heart['target'])


# In[9]:


heartdisease = len(heart[heart['target'] == 1])
no_heartdisease = len(heart[heart['target']== 0])


# In[10]:


import matplotlib.pyplot as plt
y = ('Heart Disease', 'No Disease')
y_pos = np.arange(len(y))
x = (heartdisease, no_heartdisease)
labels = 'Heart Disease', 'No Disease'
sizes = [heartdisease, no_heartdisease]
fig, ax = plt.subplots()
ax.pie(sizes,  labels=labels,  startangle=100,autopct='%1.1f%%') 
plt.title('target percentage', size=20)
plt.show() 


# In[11]:


sns.stripplot(heart['target'],heart['thalach'])


# In[12]:


import seaborn as sns
corrmat = heart.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(heart[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.title('features', size=25)
plt.show() 


# In[13]:


sns.pairplot(heart[['cp','trestbps','chol','restecg']])


# In[14]:


yes = len(heart[heart['exang'] == 1])
no = len(heart[heart['exang']== 0])


# In[15]:


import matplotlib.pyplot as plt
y = ('yes', 'no')
y_pos = np.arange(len(y))
x = (yes, no)
labels = 'yes', 'no'
sizes = [yes, no]
fig, ax = plt.subplots()
ax.pie(sizes,  labels=labels,  startangle=100,autopct='%1.1f%%') 
plt.title('exercise induced angina', size=20)
plt.show() 


# In[16]:


sns.pairplot(heart[['oldpeak','slope','exang']])


# In[19]:


from sklearn.model_selection import train_test_split
y = heart['target']
X = heart.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# # KNeighborsClassifier

# In[20]:


from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())
    


# In[21]:


knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)


# In[22]:


score.mean()


# #  RandomForestClassifier

# In[23]:


from sklearn.model_selection import cross_val_score
rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))


# In[24]:


rf_classifier= RandomForestClassifier(n_estimators=100)

score=cross_val_score(rf_classifier,X,y,cv=10)


# In[25]:


score.mean()


# In[ ]:




