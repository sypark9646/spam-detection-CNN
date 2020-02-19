from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import pandas as pd 

from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras
from time import time
import pandas as pd 

# Load the Pandas libraries with alias 'pd' 

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
train_df = pd.read_csv('C:/Users/soyeon/Downloads/training_sample.csv', encoding='utf-8') 
# Preview the first 5 lines of the loaded data 
train_df.drop(['id'], 1, inplace=True)
train_df.drop(['year_month'], 1, inplace=True)
train_df.head()

train_sample = train_df[train_df['smishing'] == 0].head(100) 
train_sample = train_sample.append(train_df[train_df['smishing'] == 1].head(100) , ignore_index=True)


from collections import Counter
import kss
word_counter = Counter()

maxlen=0
for i, row in train_df.iterrows(): 
    wordlen=0
    train_text=kss.split_sentences(row['text'])
    for sent in train_text:
        #print(i, sent)
        sent=sent.replace('X', ' ')
        word_counter.update(sent.split())
        wordlen += len(sent.split())
    maxlen=max([maxlen, wordlen])
    
vocab=dict()
voca_index=3
vocab['PAD']=0
vocab['SEN_START']=1
vocab['SEN_END']=2
for key in word_counter.keys():
    vocab[key]=voca_index
    voca_index += 1
    
print(maxlen)
print(len(vocab))



from nltk.tokenize import word_tokenize
import kss

train_df['word2ind'] = np.nan
train_df['word2ind'] = train_df['word2ind'].astype(object) 

train_df['labels'] = np.nan
train_df['labels'] = train_df['labels'].astype(object) 


for i, row in train_df.iterrows(): 
    msg=kss.split_sentences(row['text'])
    words=[]
    word2ind=[]
    for sent in msg:
        sent=sent.replace('X', ' ')
        words.extend(sent.split())
    
    for word in words:
        if word in vocab:
            word2ind.append(vocab[word])
        else:
            word2ind.append(0)

    #padding maxlen
    #a = word2ind
    #pad_value = 0
    #pad_size = maxlen - len(a)
    #word2ind = [*a, *[pad_value] * pad_size]

    train_df.at[i, 'word2ind'] = word2ind
    
    if row['smishing'] == 0: 
        train_df.at[i, 'labels'] = [1,0]
    else:
        train_df.at[i, 'labels'] = [0,1]
    

#Make Data
train_df.drop(['text'], 1, inplace=True)
train_df.drop(['smishing'], 1, inplace=True)
print(len(train_df))
train_df.head()

MAX_WORDS_IN_SEQ = 244 #
EMBED_DIM = 200
MODEL_PATH = "./models/smishing_detect"

import os
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load Data

#with open("smishing_training.pkl", 'rb') as f:
#    word2index, labels = pickle.load(f) #한줄만 읽음

data = []
labels = []

#train_df.head()

for i, row in train_df.iterrows():
    d = row['word2ind']
    l = row['labels']
    
    d = to_categorical(d)
    d = sequence.pad_sequences(d, maxlen=3288, padding='post', value=0)
    data.append(d)
    labels.append(l)
    #print(d, l)
    #break

data = np.array(data)
labels = np.array(labels)

data = sequence.pad_sequences(data, maxlen=MAX_WORDS_IN_SEQ, padding='post', value=0)
print(data.shape)
print(labels.shape)

with open("C:/Users/soyeon/Downloads/smishing_training_real.pkl", 'wb') as f:
    pickle.dump([data, labels], f) # 단 한줄씩 읽어옴