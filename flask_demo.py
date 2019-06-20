# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:48:36 2019

@author: t-baagra
"""
import numpy as np 
import flask 
import io 
import nltk
import numpy as np
import pickle
from DocSim import DocSim
import json
# nltk.download('stopwords')
# stopwords = stopwords.words("english")
stopwords=['the']
from gensim.models import Word2Vec, Phrases, phrases, KeyedVectors
import io
import requests
import pandas as pd
from flask import Flask,abort,jsonify, request
from wtforms import Form, StringField


ds=pickle.load(open(r"./model.pkl","rb"))
app=Flask(__name__)

class inputForm(Form):
    errorMessage = StringField('errorMessage')
    keyword = StringField('keyword')

@app.route('/api',methods=['POST'])
def make_predict1():
    form = inputForm(request.form)
    errorMessage = form.errorMessage.data   #   "errorMessage" yeh key hoga postman mein
    keyword = form.keyword.data     #   "keyword" yeh key hoga postman mein
    source_doc = "Qualify lead button not visible"
    source_doc=source_doc.lower()
    source_doc=source_doc.strip()
    
    #remove stopwords
    
    source_doc= ' '.join([word for word in source_doc.split() if word not in stopwords])
    
    #remove strings containing numbers
#     my_list=source_doc.split(" ")
#     my_list=[x for x in my_list if not any(c.isdigit() for c in x)]
#     source_doc=' '.join(my_list)
    
    #remove symbols and punctuations
    import string
    #source_doc=source_doc.translate(None, string.punctuation)
    source_doc=source_doc.translate(str.maketrans('','',string.punctuation))
    
    #Dynamically importing the data from blob
    #data_url='https://icmproject.blob.core.windows.net/democontainer/dataset_' + keyword + '.json'
    #response = requests.get(data_url)
    #dict=response.json()
    #target_list=dict['Rows'] #list of lists [['foo1', 'bar1'],['foo2', 'bar2']]
    
    
    url="https://icmproject.blob.core.windows.net/similarincidentsscore/ql_icm.csv"
    s=requests.get(url).content
    new_df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    
    #Remove NANs
    col=['IncidentId','Error Message']
    new_df=new_df[col]
    new_df.columns=['IncidentId','Error Message']
    new_df=new_df.dropna()
  

    
    #######################################################################Preprocessing script################################################################
        
    #Remove strings with numbers (Problem)
#     for idx,row in new_df.iterrows():
#         my_list=[]
#         my_list=row['Error Message'].split(" ")
#         my_list=[x for x in my_list if not any(c.isdigit() for c in x)]
#         row['Error Message']=' '.join(my_list)
    
    #Removing urls
    new_df['Error Message'] = new_df['Error Message'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)    
        
    #Remove symbols and spaces
    import string
    new_df['Error Message']=new_df['Error Message'].str.replace(r"\[.*\]","")
    new_df['Error Message'] = new_df['Error Message'].str.replace('[{}]'.format(string.punctuation), '')    
    
#     #Remove Stopwords (Problem) 
#     new_df['Error Message']=new_df['Error Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    
    #Lowercase
    new_df['Error Message']=new_df['Error Message'].str.lower()
    ###################################################################################################################################################################
    
    
    target_list=new_df.reset_index()[['IncidentId','Error Message']].values.tolist()  #list of lists [['1234567', 'Qualify lead button not visible'],['foo2', 'bar2']]
    
    
    #Converting into dictionary {'key':[]} that maps ErrorMessage to List of Incident Ids
    import collections
    d=collections.defaultdict(list)
    for item in target_list:
        d[item[1]].append(item[0])
        
    
    target_docs=new_df['Error Message'].values.tolist()
    
    ######################################################################Calculate similarity#################################################################
   # words = [w for w in source_doc.split(" ") if w not in stopwords]
   # word_vecs = []
   # for word in words:
   #     try:
   #         vec = model[word]
   #         word_vecs.append(vec)
   #     except KeyError:
   #         # Ignore, if the word doesn't exist in the vocabulary
   #         pass
   # source_vec=np.mean(word_vecs,axis=0)#

    #results=[]
   # for doc in target_docs:
   #     words=[w for w in doc.split(" ") if w not in stopwords]
   #     word_vecs=[]
   #     for word in words:
   #         try:
   #             vec=model[word]
   #             word_vecs.append(vec)

  #          except KeyError:
   #             pass
   #     target_vec=np.mean(word_vecs,axis=0)
   #     csim=np.dot(source_vec, target_vec) / (np.linalg.norm(source_vec) * np.linalg.norm(target_vec))
   #     if np.isnan(np.sum(csim)):
   #         csim=0
   #     results.append({
   #                     'score' : csim,
   #                     'doc' : doc
   #                 })
   # results.sort(key=lambda k : k['score'] , reverse=True)
    
    sim_scores = ds.calculate_similarity(source_doc, target_docs)
    sim_scores=sim_scores[:5]
    final_list={000000000.0}
    for l in sim_scores:
        for item in d[l['doc']]:
            final_list.add(item)
    
    ans=[]
    for item in final_list:
        if(item != 0.0):
            ans.append(item)
    my_json = json.dumps(ans)
    return my_json


if __name__=='__main__':
    app.run(debug=True,use_reloader=False)