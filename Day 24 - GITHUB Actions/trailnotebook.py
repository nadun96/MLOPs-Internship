#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


import matplotlib.pyplot as plt


# In[6]:


data = pd.read_csv("heart.csv")


# In[8]:


data.head()


# In[9]:


data.info()


# In[10]:


data.isnull().sum()


# In[11]:


data.describe()


# In[12]:


data.head()


# In[13]:


sns.kdeplot(data.oldpeak)
plt.show()


# In[14]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
mmscaler = MinMaxScaler()
sscaler = StandardScaler()


# In[15]:


data[["chol", "oldpeak", "thalach"]] = mmscaler.fit_transform(data[["chol", "oldpeak", "thalach"]])


# In[16]:


data.head()


# In[17]:


data[["age", "trestbps"]] = sscaler.fit_transform(data[["age", "trestbps"]])


# In[18]:


data.head()


# In[19]:


x = data.drop(["target"], axis=1)
y = data.target


# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[21]:


x_train.shape


# In[22]:


y_train.shape


# In[23]:


x_test.shape


# In[24]:


y_test.shape


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


model1 = LogisticRegression()


# In[27]:


model1.fit(x_train, y_train)


# In[28]:


answer1 = model1.predict(x_test)


# In[29]:


x_test.shape


# In[30]:


answer1


# In[31]:


from sklearn.svm import SVC
model2 = SVC()


# In[32]:


model2.fit(x_train, y_train)


# In[33]:


answer2 = model2.predict(x_test)


# In[34]:


answer2


# In[35]:


from sklearn.metrics import accuracy_score


# In[36]:


score_lr = accuracy_score(y_test, answer2)
print(score_lr*100)


# In[46]:


test_score = open("test.txt", "w")
test_score.write(f"Test score: %2.1f%%" % score_lr)


# In[38]:


from sklearn.metrics import accuracy_score
train_ans = model2.predict(x_train)
training_result = accuracy_score(y_train, train_ans)
training_result


# In[47]:


train_score = open("train.txt", "w")
train_score.write(f"Training score: %2.1f%%" % training_result)


# In[40]:


score_svm = accuracy_score(y_test, answer2)
print(score_svm*100)


# In[41]:


from sklearn.metrics import classification_report
score_svm = classification_report(y_test, answer2)
print(score_svm)


# In[42]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_train, train_ans)


# In[43]:


print(matrix)


# In[44]:


sns.heatmap(matrix, annot=True)


# In[45]:


fig,ax = plt.subplots()
ax = sns.heatmap(matrix, annot=True, fmt=".1f")
plt.tight_layout()
plt.savefig("heatmap.png",dpi=120) 
plt.close()

