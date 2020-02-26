# -*- coding: utf-8 -*- 
import fasttext

from keras.utils import to_categorical
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import pandas as pd 
import kss
import os
from fasttext_dataset import LoadDataset

ft_model = fasttext.load_model('cc.ko.300.bin')

MODEL_PATH = "./models/smishing_detect_fasttext"
data_path = "C:/Users/soyeon/Downloads/training_sample.csv"
model = load_model(MODEL_PATH)

# Data setup.
dataset = LoadDataset(csv_path=data_path)

MAX_WORDS_IN_SEQ = dataset.MAX_WORDS_IN_SEQ
EMBED_DIM = 300

#Prediction
test_text = "믿고 거래해주셔서 진심으로 감사합니다.행복한하루되세요XXX당진XXX올림"
msg=kss.split_sentences(test_text)
words=[]
word2ind=[]
for sent in msg:
    sent=sent.replace('X', ' ')
    words.extend(sent.split())
for word in words:
    word2ind.append(list(ft_model.get_word_vector(word)))
for padding in range(dataset.MAX_WORDS_IN_SEQ-len(words)):
    word2ind.append([0.0 for padding in range(300)])
word2ind=np.array(word2ind).reshape(1, MAX_WORDS_IN_SEQ, 300)

print(word2ind.shape)

prediction = model.predict({'input_1': word2ind })
print(prediction)

if prediction[0][0]>prediction[0][1]:
    print(0)
else:
    print(1)
