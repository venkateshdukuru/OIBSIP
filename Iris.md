# OIBSIP

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[7]:


df=pd.read_csv('F:/Iris.csv')
df.head()


# In[8]:


df.info()


# In[9]:


df.shape


# In[10]:


df.columns


# In[11]:


df.nunique()


# In[15]:



df.head()


# In[17]:


df["Species"].replace({"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3},inplace=True)


# In[18]:


df


# In[19]:


x=pd.DataFrame(df,columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]).values


# In[20]:


x


# In[22]:


y=df.Species.values.reshape(-1,1)


# In[23]:


y


# In[24]:

from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)


# In[27]:


x_train.shape


# In[29]:


y_train.shape


# In[41]:


metrics.accuracy_score(y_test,y_pred)*100







