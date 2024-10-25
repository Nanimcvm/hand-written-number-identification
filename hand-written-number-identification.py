#!/usr/bin/env python
# coding: utf-8

# In[27]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as P
import matplotlib.pyplot as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as npy


# In[33]:


(X_train,Y_train),(X_test,Y_test)=keras.datasets.mnist.load_data()


# In[6]:


len(X_train)


# In[7]:


len(X_test)


# In[8]:


X_train[0]


# In[14]:


X_train[7].shape


# In[13]:


mpl.matshow(X_train[5])


# In[15]:


mpl.matshow(X_train[8])


# In[20]:


Y_train[:9]


# In[17]:


X_train.shape


# In[18]:


X_train=X_train/255
X_test=X_test/255


# In[19]:


X_train[2]


# In[21]:


X_train_flattened=X_train.reshape(len(X_train),28*28)
X_test_flattened=X_test.reshape(len(X_test),28*28)


# In[22]:


X_train_flattened.shape


# In[23]:


X_train_flattened[2]


# In[52]:


input_layer = Input(shape=(28*28))
hidden_layer = Dense(500, activation='sigmoid')(input_layer)
output_layer = Dense(20, activation='softmax')(hidden_layer)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_flattened, Y_train, epochs=20)


# In[53]:


model.evaluate(X_test_flattened,Y_test)


# In[54]:


Y_predicted=model.predict(X_test_flattened)
Y_predicted[2]


# In[55]:


model.predict(X_test_flattened)


# In[56]:


mpl.matshow(X_test[6])


# In[57]:


npy.argmax(Y_predicted[0])


# In[58]:


Y_predicted[0]


# In[59]:


mpl.matshow(X_test[50])


# In[60]:


Y_predicted[50]


# In[61]:


npy.argmax(Y_predicted[50])


# In[62]:


mpl.matshow(X_test[30])


# In[63]:


npy.argmax(Y_predicted[30])


# In[64]:


Y_predicted[30]


# In[65]:


mpl.matshow(X_train[5])


# In[66]:


mpl.matshow(X_train[7])


# In[ ]:




