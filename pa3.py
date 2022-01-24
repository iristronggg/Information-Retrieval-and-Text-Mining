#!/usr/bin/env python
# coding: utf-8

# In[208]:


from nltk.stem.porter import PorterStemmer
import os
import string
import pandas as pd
import math
#from scipy import spatial
import numpy as np


# In[209]:


def readTraining(filename):
    classDict={}
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.split()
            classDict[item[0]] = item[1:len(item)]

    return(classDict)


# In[ ]:


trainDic = readTraining("training.txt")
# print(trainDic)
# print(sum(trainDic.values(),[]))


# In[ ]:


def getTerms(filename):
    cnt = 0
    with open("data/"+filename+".txt", 'r') as f: # open in readonly mode
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

    return final_terms


# In[ ]:


##所有檔案前處理
def extractVocab(fileList):
    cnt = 0
    for filename in fileList:
        with open("data/"+filename+".txt", 'r') as f: # open in readonly mode
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
            # docTerms.insert(1,'df',0)
            
        else:
            tempTerms = pd.DataFrame(final_terms,columns=['term'])
            tempTerms = tempTerms.groupby('term').size().reset_index(name = 'tf')

            docTerms = pd.concat([docTerms, tempTerms]).groupby('term').sum().reset_index()
            docTerms = docTerms.sort_values(by=['term'])
            # docTerms = docTerms.merge(tempTerms, on="term", how="outer")
            # docTerms['df'].fillna(value=0, inplace=True)
            # docTerms['df'] += docTerms.count(axis = "columns").values - 2 
            # # term and df left 
            # docTerms.drop(docTerms.columns[2:4],axis=1,inplace=True)

    # docTerms = docTerms.sort_values(by=['term'])
    # docTerms['df'] = docTerms['df'].astype(int)
    # docTerms.insert(0,'t_index',0)
    # docTerms['t_index'] = range(1, docTerms.shape[0]+1)
    # docTerms.reset_index()
    print("Dictionary created.")
    return docTerms


# In[221]:


classCnt = 13
classOfDocTok = list()
for classN in trainDic.values():
    classTok = list()
    for doc in classN:
        tokens = getTerms(doc)
        classTok.append(tokens)
    classOfDocTok.append(classTok)
    
trainDocList = sum(trainDic.values(),[])
vocab = extractVocab(trainDocList)
for i in range(classCnt):
    vocab['class'+str(i+1)]=0

for t in vocab['term']:
    for i in range(classCnt): 
        n11=0
        n10=0
        n01=0
        n00=0
        for docTok in classOfDocTok[i]: # on topic
            if t in docTok:
                n11 += 1
            else:
                n10 += 1

        for j in range(classCnt): #off topic
            if j != i:
                for docTok in classOfDocTok[j]:
                    if t in docTok:
                        n01 += 1
                    else:
                        n00 += 1
        N = n00+n01+n10+n11
        pt = (n11+n01) / N
        p1 = n11 / (n11+n10)
        p2 = n01 / (n01+n00)
        L = ((pt**n11)*((1-pt)**n10)*(pt**n01)*((1-pt)**n00)) / ((p1**n11)*((1-p1)**n10)*(p2**n01)*((1-p2)**n00))
        LR=(-2)*math.log2(L)
        vocab.loc[vocab['term']==t,'class'+str(i+1)] = LR
    
    # print(vocab[vocab['term']==t])
    
        


# In[212]:


ave_LR = vocab.iloc[:,2:15].mean(axis=1)
vocab['ave_LR'] = ave_LR
vocab = vocab.sort_values(by=['ave_LR'], ascending=False)
# print(vocab)
final_vocab = vocab.iloc[0:500,:]
print(final_vocab)


# In[214]:


def countTF(docList, vocabulary):
    docVoc = extractVocab(docList)
    docVoc = docVoc.merge(vocabulary, on="term", how="right")
    docVoc.fillna(value=0, inplace=True)
    # print(docVoc['tf'].sum())
    docVoc['prob'] = (docVoc['tf']+1)/((docVoc['tf'].sum()+len(docVoc)))
    docVoc = docVoc.drop(columns=['tf'])
    # print(docVoc)
    return docVoc
    


# In[215]:


def trainMultinomialNB(trainClass,final_vocab):
    prior = {}
    trainDocList = sum(trainClass.values(),[])
    # vocab = extractVocab(trainDocList)
    # vocab = final_vocab
    # print(vocab)
    vocab_term = final_vocab['term']
    condprob = pd.DataFrame(vocab_term,columns=['term'])
    N_doc = len(trainDocList)
    for classKey in trainClass:
        N_c = len(trainClass[classKey])
        prior[classKey] = N_c / N_doc
        classTF = countTF(trainClass[classKey], vocab_term)
        # print(classTF)
        classTF.rename(columns={'prob': 'class_'+classKey}, inplace=True)
        # print(classTF)
        condprob = condprob.merge(classTF, on="term", how="left")
        # condprob.fillna(value=0, inplace=True)

    return vocab_term, prior, condprob
        




# In[216]:


train_class = readTraining("training.txt")
train_vocab, prior, condprob = trainMultinomialNB(train_class,final_vocab)
print(condprob)
# selectFeatures()


# In[217]:


train_list = sum(trainDic.values(),[])
all_list = list(range(1,1096))
all_list = [str(x) for x in all_list]
test_list = list(set(all_list)-set(train_list))
# print(test_list)


# In[218]:


result = {}
for filename in range(1,1096):
    if str(filename) not in train_list:
        test_vocab = getTerms(str(filename))
        score_list = []
        for classN in range(1,classCnt+1):
            score_c = math.log2(prior[str(classN)])
            for t in test_vocab:
                if t in list(train_vocab):
                    # print(condprob.loc[condprob['term']==t,'class_'+str(classN)].values[0])
                    score_c += math.log2(condprob.loc[condprob['term']==t,'class_'+str(classN)].values[0])
            score_list.append(score_c)
        # print(score_list)
        result[filename] = score_list.index(max(score_list))+1


# In[219]:


# print(result)


# In[220]:


with open('result.csv', 'w') as f:
	f.write("%s,%s\n"%('Id','Value'))
	for key in result.keys():
		f.write("%s,%s\n"%(key,result[key]))
f.close()

