# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:38:24 2019

@author: t-baagra
"""
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import tokenize

from gensim.models import Word2Vec, Phrases, phrases, KeyedVectors
from DocSim import DocSim

import io
import requests
url="https://icmproject.blob.core.windows.net/similarincidentsretrain/title.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))
print(df)

#Remove strings with numbers
for idx,row in df.iterrows():
    my_list=[]
    my_list=row['Title'].split(" ")
    my_list=[x for x in my_list if not any(c.isdigit() for c in x)]
    row['Title']=' '.join(my_list)
    
#Remove stopwords
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')    
df['Title']=df['Title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Remove symbols
import string
import pandas as pd
df['Title']=df['Title'].str.replace(r"\[.*\]","")
df['Title'] =df['Title'].str.replace('[{}]'.format(string.punctuation), '')

#Lowercasing
df['Title']=df['Title'].str.lower() 
df['Title']  
################################################################################################
#Getting error messages
df3=pd.read_excel('QL_ICM_data.xlsx')
col=['Error Message']
df3=df3[col]
df3.columns=['Error Message']
df3=df3.dropna()

#Remove strings with numbers
for idx,row in df3.iterrows():
    my_list=[]
    my_list=row['Error Message'].split(" ")
    my_list=[x for x in my_list if not any(c.isdigit() for c in x)]
    row['Error Message']=' '.join(my_list)
    
#Remove stopwords
stop = stopwords.words('english')    
df3['Error Message']=df3['Error Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Remove symbols
df3['Error Message']=df3['Error Message'].str.replace(r"\[.*\]","")
df3['Error Message'] =df3['Error Message'].str.replace('[{}]'.format(string.punctuation), '')

#Lowercasing
df3['Error Message']=df3['Error Message'].str.lower() 
df3['Error Message'] 
##########################################################################################################
nltk.download('punkt')

#Getting the preprocessed summary
url="https://icmproject.blob.core.windows.net/similarincidentsretrain/preprocessed_summary.csv"
s1=requests.get(url).content
df1=pd.read_csv(io.StringIO(s1.decode('utf-8')))
df1




#Getting the preprocessed comments
url="https://icmproject.blob.core.windows.net/similarincidentsretrain/preprocessed_comments.csv"
s2=requests.get(url).content
df2=pd.read_csv(io.StringIO(s2.decode('utf-8')))
df2


def custom_tokenize(text):
    if not text:
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return word_tokenize(text)


#Tokenizing title
df['tokenized'] = df.apply(lambda row: nltk.word_tokenize(row['Title']), axis=1)

#Tokenizing error
df3['tokenized']=df3.apply(lambda row: nltk.word_tokenize(row['Error Message']), axis=1)

#Tokenizing Summary
df1=df1.dropna()
df1['tokenized']=df1['Preprocessed Summary'].apply(custom_tokenize)

#Tokenizing Comments
df2=df2.dropna()
df2['tokenized']=df2['Preprocessed Text'].apply(custom_tokenize)


#Concatenating the Title and summary and comments
df4=pd.concat([df['tokenized'],df3['tokenized'],df1['tokenized'],df2['tokenized']],ignore_index=True,join='outer')
df4

model = Word2Vec(size=300, min_count=5, iter=10)
model.build_vocab(df4)
training_examples_count = model.corpus_count

model.train(df4,total_examples=training_examples_count, epochs=model.iter)

with open('stopwords_en.txt', 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

import pickle
pickle.dump(ds,open(r"C:\Users\t-baagra\Desktop\ChillDev\Duplicate questions\model.pkl","wb"))
