from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd 
import kss
from nltk.tokenize import word_tokenize
import fasttext

class LoadDataset():
    def __init__(self, csv_path): #download, read data 등등을 하는 파트
        '''
        windows 에서 fasttext는
        pip install cython
        pip install fasttext-win 설치 후 사용
        '''
        #fasttext.download_model('ko', if_exists='ignore')  # Korean
        ft_model = fasttext.load_model('cc.ko.300.bin')

        csv_path = csv_path
        train_df= pd.read_csv(csv_path, encoding='utf-8') 

        train_df['word2ind'] = np.nan
        train_df['word2ind'] = train_df['word2ind'].astype(object)
        train_df['labels'] = np.nan
        train_df['labels'] = train_df['labels'].astype(object) 

        maxlen=0
        for i, row in train_df.iterrows(): 
            wordlen=0
            train_text=kss.split_sentences(row['text'])
            for sent in train_text:
                sent=sent.replace('X', ' ')
                wordlen += len(sent.split())
            maxlen=max([maxlen, wordlen])
        
        self.MAX_WORDS_IN_SEQ = maxlen

        for i, row in train_df.iterrows(): 
            msg=kss.split_sentences(row['text'])
            words=[]
            word2ind=[]
            for sent in msg:
                sent=sent.replace('X', ' ')
                words.extend(sent.split())
            
            for word in words:
                word2ind.append(list(ft_model.get_word_vector(word)))
            
            for padding in range(self.MAX_WORDS_IN_SEQ-len(words)):
                word2ind.append([0.0 for padding in range(300)])
            #print(word2ind)
            #print(i)
            train_df.at[i, 'word2ind'] = word2ind
            
            if row['smishing'] == 0: 
                train_df.at[i, 'labels'] = [1,0]
            else:
                train_df.at[i, 'labels'] = [0,1]

        #Make Data
        train_df.drop(['text'], 1, inplace=True)
        train_df.drop(['smishing'], 1, inplace=True)
        
        self.data = []
        self.labels = []
        for i, row in train_df.iterrows():
            d = row['word2ind']
            l = row['labels']
            self.data.append(d)
            self.labels.append(l)

        del train_df
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        #print(self.data.shape, self.labels.shape)

    def MAX_WORDS_LENGTH(self): 
        return self.MAX_WORDS_IN_SEQ

    def DATA_AND_LABEL(self):
        return self.data, self.labels