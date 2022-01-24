#!/usr/bin/env python
# coding: utf-8

# In[57]:


from nltk.stem.porter import PorterStemmer
import os
import string
import pandas as pd
import math
#from scipy import spatial
import numpy as np


# In[58]:


df2 = pd.DataFrame(columns = ['df'])
# df2


# In[59]:


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
        docTerms['df'].fillna(value=0, inplace=True)  
        docTerms['df'] += docTerms.count(axis = "columns").values - 2 
        # term and df left 
        docTerms.drop(docTerms.columns[2:4],axis=1,inplace=True) 

            
    # final_terms = list(set(final_terms))
    # final_terms = sorted(final_terms)
    
    # for t in final_terms:
    #     if (df2.index == t).any():
    #         df2.loc[t] += 1
    #     else:
    #         df2.loc[t] = 1
    
    # all_terms += [t for t in final_terms if not t in all_terms]
    # all_terms = sorted(all_terms)


# In[60]:


docTerms = docTerms.sort_values(by=['term'])
docTerms['df'] = docTerms['df'].astype(int)
docTerms.insert(0,'t_index',0)
docTerms['t_index'] = range(1, docTerms.shape[0]+1)
print("Dictionary created.")


# In[61]:


#編號
# df2['t_index']=range(1, len(df2)+1)
# df2
# docTerms


# In[62]:


#輸出結果
docTerms.to_csv(r'dictionary.txt', header=True, index=False, sep=' ', mode='w')
# with open('dictionary.txt', 'w') as f:
#     for row in df2.index:
#         df = df2.loc[row]['df']
#         i = df2.loc[row]['t_index']
#         f.write("%d %s %s\n" %(i,row,df))


# In[63]:


tempTerms = pd.DataFrame()


# In[64]:


def fileToVec(filename):
    with open(os.path.join("data", filename), 'r') as f: # open in readonly mode
        data = f.read().replace('\n', '') #讀檔與去除換行符號

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
    tempTerms.insert(1,'tf-idf',0) 

    tempTerms = tempTerms.merge(docTerms, how="left", on="term")
    tempTerms['tf-idf'] = tempTerms['tf'] * np.log10(1095/tempTerms['df'])
    tempTerms.drop(columns=['term', 'tf', 'df'],inplace = True)
    tempTerms = tempTerms[[ 't_index' , 'tf-idf']]
    tempTerms = tempTerms.sort_values(by=['t_index'])
    # Unit Vector
    tempTerms['tf-idf'] = tempTerms['tf-idf'] / (tempTerms['tf-idf']**2).sum()**0.5
    
    #輸出結果
    doc_vecfile = os.path.join("output", "doc"+filename)
    f = open(doc_vecfile, "w")
    f.write(str(tempTerms.shape[0])+"\n")
    f.close()
    tempTerms.to_csv(doc_vecfile, header=True, index=False, sep=' ', mode='a')
    


# In[65]:


## All doc to vector
for filename in os.listdir("data"):
    tempTerms.drop(tempTerms.index, inplace=True)
    fileToVec(filename)

    


# In[66]:


def readDocToVec(filename):
    t_index=[]
    unit_tf_idf=[]
    with open("output/doc"+filename,'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            item = line.split()
            t_index.append(int(item[0]))
            unit_tf_idf.append(float(item[1]))

    docVec = pd.DataFrame(list(zip(t_index,unit_tf_idf)),columns=['t_index','unit_tf_idf'])
    return(docVec)

            


# In[67]:


###### Cosine Similarity ######
def cosine(x,y):
    fileToVec(x)
    fileToVec(y)
    
    Vec1 = readDocToVec(x)
    Vec2 = readDocToVec(y)
    Vec = Vec1.merge(Vec2, how="inner", on="t_index")
    print(Vec)

    #算相似度
    sim = np.dot(Vec["unit_tf_idf_x"], Vec["unit_tf_idf_y"])
    print("\nCosine Similarity: ",sim)


# In[68]:


#計算doc1.doc2的相似度
cosine("1.txt","2.txt")

