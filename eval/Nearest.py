#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle


# In[2]:


mus = np.load("./mu.npy")


# In[3]:


mus.dtype


# In[4]:


mus = np.nan_to_num(mus)


# In[5]:


np.any(np.isnan(mus))


# In[6]:


word_dictionary = pickle.load(open("data.pkl","rb"),encoding="latin1")


# In[ ]:





# In[7]:


reverse_word_dictionary = {value:key for key,value in zip(word_dictionary.keys(), word_dictionary.values())}


# In[8]:


len(word_dictionary)


# In[9]:


nbrs = NearestNeighbors(n_neighbors=50, algorithm='brute').fit(mus)


# In[10]:


#look up word here call it x_not
x_index = word_dictionary["fat"]
x_not = mus[x_index].reshape(1,-1)


# In[11]:


x_not.shape


# In[12]:


distance, indicies = nbrs.kneighbors(x_not)


# In[13]:


for i in indicies[0]: 
    print(reverse_word_dictionary[i])


# In[ ]:




