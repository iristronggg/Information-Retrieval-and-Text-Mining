#!/usr/bin/env python
# coding: utf-8

# In[20]:


from nltk.stem.porter import PorterStemmer


# In[21]:


with open("PA1text.txt", "r") as f:    #開啟檔案
    data = f.read().replace('\n', '') #讀檔與去除換行符號
    print(data)


# In[22]:


#去除標點
data = data.replace(',','')
data = data.replace('.','')
data = data.replace('\'','')
data


# In[23]:


#lowercase, tokenize
data = data.lower()
tokens = data.split()
print(len(tokens))
print(tokens)


# In[24]:


#use Porter's Stemmer
stemmer = PorterStemmer()


# In[25]:


terms = [stemmer.stem(token) for token in tokens]
print(len(terms))
print(terms)


# In[26]:


with open("stop_words.txt", "r") as f:    #開啟stop_words.txt
    stop_words = f.read() #讀檔
    stop_words = stop_words.split()
    #print(stop_words)


# In[27]:


#去除stopwords
#stop_words = set(stopwords.words('english')) 
final_terms = [w for w in terms if not w in stop_words]
print(len(final_terms))
print(final_terms)


# In[28]:


#輸出結果
with open('result.txt', 'w') as f:
    for term in final_terms:
        f.write("%s\n" % term)


# In[ ]:




