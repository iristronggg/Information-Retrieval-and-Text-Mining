#!/usr/bin/env python
# coding: utf-8

# In[73]:


from nltk.stem.porter import PorterStemmer
import os
import string
import pandas as pd
import math
import numpy as np
from os import listdir




# ## Create Dictionary

# In[37]:
doc_size = 1095

all_terms = []
##所有檔案前處理
cnt = 0
for filename in os.listdir("data"):
    with open(os.path.join("data", filename), 'r') as f: # open in readonly mode
        data = f.read().replace('\n', '') #讀檔與去除換行符號
        
    cnt += 1
    #去除標點與數字
    for c in string.punctuation:
        data = data.replace(c,"")
    for i in range(10):
        data = data.replace(str(i),"")
        
    #lowercase, tokenize
    data = data.lower()
    tokens = data.split()

    #use Porter's Stemmer
    stemmer = PorterStemmer()
    terms = [stemmer.stem(token) for token in tokens]

    # with open("stop_words.txt", "r") as f:    #開啟stop_words.txt
    #     stop_words = f.read() #讀檔
    #     stop_words = stop_words.split()
    stop_words = ['a', "a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there's", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']

    #去除stopwords
    #stop_words = set(stopwords.words('english'))
    final_terms = []
    final_terms1 = [w for w in terms if not w in stop_words]
    for terms in final_terms1:
        if 'http' in terms or 'www' in terms:
            continue
        else:
            final_terms.append(terms)
    
    if cnt == 1:
        docTerms = pd.DataFrame(final_terms,columns=['term'])
        docTerms = docTerms.groupby('term').size().reset_index(name = 'tf')
        docTerms.insert(1,'df',0)
        
    else:
        tempTerms = pd.DataFrame(final_terms,columns=['term'])
        tempTerms = tempTerms.groupby('term').size().reset_index(name = 'tf')

        docTerms = docTerms.merge(tempTerms, on="term", how="outer") 
        # print(docTerms)
        docTerms['df'].fillna(value=0, inplace=True)  
        docTerms['df'] += docTerms.count(axis = "columns").values - 2 
        # term and df left 
        # print(docTerms)
        docTerms.drop(docTerms.columns[2:4],axis=1,inplace=True) 

            
    # final_terms = list(set(final_terms))
    # final_terms = sorted(final_terms)
    
    # for t in final_terms:
    #     if (df2.index == t).any():
    #         df2.loc[t] += 1
    #     else:
    #         df2.loc[t] = 1
    
    # all_terms += [t for t in final_terms if not t in all_terms]
    # all_terms = sorted(all_terms)docTerms = docTerms.sort_values(by=['term'])

docTerms = docTerms.sort_values(by=['term'])    
docTerms['df'] = docTerms['df'].astype(int)
docTerms.insert(0,'t_index',0)
docTerms['t_index'] = range(1, docTerms.shape[0]+1)
print("Dictionary created.")
print(docTerms)


# In[39]:


vec_size = len(docTerms)


# In[116]:


def fileToVec(filename):
    with open(os.path.join("data", filename), 'r') as f: # open in readonly mode
        data = f.read().replace('\n', '') #讀檔與去除換行符號

    tempTerms = pd.DataFrame()

    # print(filename)
    #去除標點與數字
    for c in string.punctuation:
        data = data.replace(c,"")
    for i in range(10):
        data = data.replace(str(i),"")

    #lowercase, tokenize
    data = data.lower()
    tokens = data.split()

    #use Porter's Stemmer
    stemmer = PorterStemmer()
    terms = [stemmer.stem(token) for token in tokens]

    #去除stopwords
    stop_words = ['a', "a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there's", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']

    final_terms = []
    final_terms1 = [w for w in terms if not w in stop_words]
    for terms in final_terms1:
        if 'http' in terms or 'www' in terms:
            continue
        else:
            final_terms.append(terms)

    tempTerms = pd.DataFrame(final_terms,columns=['term'])
    tempTerms = tempTerms.groupby('term').size().reset_index(name = 'tf')
    tempTerms.insert(2,'tf-idf',0) 
    # print(tempTerms)

    tempTerms = tempTerms.merge(docTerms, how="right", on="term")
    tempTerms['tf'].fillna(value=0, inplace=True)
    tempTerms['tf-idf'] = tempTerms['tf'] * np.log10(1095/tempTerms['df'])
    # print(tempTerms)
    tempTerms.drop(columns=['term', 'tf', 'df'],inplace = True)
    tempTerms = tempTerms[[ 't_index' , 'tf-idf']]
    tempTerms = tempTerms.sort_values(by=['t_index'])
    # Unit Vector
    tempTerms['tf-idf'] = tempTerms['tf-idf'] / (tempTerms['tf-idf']**2).sum()**0.5
    # print(tempTerms)
    print(tempTerms)
    return tempTerms


# In[153]:


docVecList = []
for i in range(1,doc_size+1):
    docVecList.append(fileToVec(str(i)+".txt")['tf-idf'])
docVec = np.asarray(docVecList)
print(docVec.shape)
print(docVec)


# In[154]:


# docVec = np.array([])
print(docVec)


# In[ ]:


sim_list_all = []
for i in range(doc_size):
    sim_list = []
    for j in range(doc_size):
        sim_list.append(np.dot(docVec[i],docVec[j]))
    sim_list_all.append(sim_list)
print(sim_list_all)


# In[185]:


sim = np.asarray(sim_list_all)
print(sim)


# In[159]:


# l = []
# for i in range(1000):
#     l.append([3*i+1,3*i+2,3*i+3])
# l = np.asarray(l)
# print(l)
np.save("docSim.npy",sim)


# In[198]:


I = [1]*doc_size
C = np.load("docSim.npy")
A = []
C.shape


# In[199]:


def find_max(I, C):
    max_sim = -1
    index_i = -1
    index_m = -1
    for x in range(doc_size):
        if I[x] == 1:
            for y in range(doc_size):
                if I[y] == 1 and x != y:
                    if C[x][y] > max_sim:
                        max_sim = C[x][y]
                        index_i = x
                        index_m = y
    # print("max============"+str(max_sim))
    return index_i, index_m


# In[200]:


for k in range(doc_size-1):
    i, m = find_max(I,C)
    A.append([i,m])
    for j in range(doc_size):
        C[i][j] = min(C[i][j],C[m][j])
        C[j][i] = min(C[i][j],C[m][j])
    I[m] = 0
    # print(k)


# In[202]:


print(A)
# print(len(A))


# In[222]:


def write_cluster(cluster,n):
    with open(str(n)+".txt",'w') as f:
        cnt = 0
        print(len(cluster))
        for val in cluster.values():
            doc_cluster = np.sort(val)
            for ele in doc_cluster:
                f.write(str(ele+1)+'\n')
            cnt += 1
            if cnt != len(cluster):
                f.write('\n')


# In[223]:


cluster = {}
for i in range(doc_size):
    cluster[str(i)] = [i]

for i,m in A:
    merge = cluster[str(m)]
    cluster.pop(str(m))
    cluster[str(i)] += merge
    if len(cluster) == 20:
        write_cluster(cluster,20)
    if len(cluster) == 13:
        write_cluster(cluster,13)
    if len(cluster) == 8:
        write_cluster(cluster,8)
    


# In[ ]:




