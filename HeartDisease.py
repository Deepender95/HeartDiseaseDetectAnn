#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/user/Downloads/archive/heart_2020_cleaned.csv")
data.head()


# In[2]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["HeartDisease"]=le.fit_transform(data["HeartDisease"])
data["Smoking"]=le.fit_transform(data["Smoking"])
data["AlcoholDrinking"]=le.fit_transform(data["AlcoholDrinking"])
data["Stroke"]=le.fit_transform(data["Stroke"])
data["DiffWalking"]=le.fit_transform(data["DiffWalking"])
data["Sex"]=le.fit_transform(data["Sex"])
data["AgeCategory"]=le.fit_transform(data["AgeCategory"])
data["Race"]=le.fit_transform(data["Race"])
data["Diabetic"]=le.fit_transform(data["Diabetic"])
data["PhysicalActivity"]=le.fit_transform(data["PhysicalActivity"])
data["GenHealth"]=le.fit_transform(data["GenHealth"])
data["Asthma"]=le.fit_transform(data["Asthma"])
data["KidneyDisease"]=le.fit_transform(data["KidneyDisease"])
data["SkinCancer"]=le.fit_transform(data["SkinCancer"])


# In[3]:


x=data.iloc[:,[1,2,3,4,5,8,9,11]]
y=data.iloc[:,0]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)


# In[4]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)


# In[5]:


from tensorflow import keras
model=keras.models.Sequential([
    keras.layers.Dense(64,activation='relu',input_shape=(8,)),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')

])


# In[6]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(xtrain,ytrain,epochs=10,batch_size=32,validation_data=(xtest,ytest))


# In[7]:


_,accuracy=model.evaluate(xtest,ytest)
print("Accuracy of model through ANN classifier is: ",accuracy*100)


# In[11]:


model.save("model.h5")

