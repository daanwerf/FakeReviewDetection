#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#Load dataset here.
#data in the shape of an array of features. [[features_sample1],[features_sample2], etc]
#target in the shape of [1 0 1 1 0 1] where 1 corresponts with the first sample.


# In[10]:


# Split dataset into training set and test set. X is input, Y is target
# replace data and target with the correct data and target.
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3,random_state=109) # 70% training and 30% test


# In[12]:


clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[13]:


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[14]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[ ]:




