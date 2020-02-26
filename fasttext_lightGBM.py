# coding: utf-8
import numpy as np
import lightgbm as lgb
import pickle
import kss
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from fasttext_dataset import LoadDataset
import fasttext
from fasttext_dataset import LoadDataset

#this code is from [Python Machine Learning Cookbook]

if __name__ == '__main__':
    
    ft_model = fasttext.load_model('cc.ko.300.bin')

    data_path = "C:/Users/soyeon/Downloads/training_sample.csv"

    print('Loading data...')
    # Data setup.
    dataset = LoadDataset(csv_path=data_path)
    MAX_WORDS_IN_SEQ = dataset.MAX_WORDS_IN_SEQ
    data, labels = dataset.DATA_AND_LABEL()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    #to reduce dimensionality
    x_train = x_train.reshape(160, 73200)
    x_test = x_test.reshape(40, 73200)
    y_trains = [a for a, b in y_train]
    y_tests = [b for a, b in y_test]
    y_train = y_trains
    y_test = y_tests
    print(x_train.shape, len(y_train), x_test.shape, len(y_test))

    # create dataset for lightgbm
    lgbTrain=lgb.Dataset(x_train, y_train)
    lgbEval=lgb.Dataset(x_test, y_test, reference=lgbTrain)

    # specify your configurations as a dict
    parameters = {
        'boosting_type':'gbdt',
        'objective':'binary',
        'metric':'binary_logloss',
        'num_leaves':31,
        'learning_rate':0.001,
        'feature_fraction':0.9,
        'bagging_fraction':0.8,
        'bagging_freq':5,
        'verbose':0
    }

    print('Starting training...')
    # train
    gbm=lgb.train(parameters, lgbTrain, num_boost_round=20, valid_sets=lgbTrain, early_stopping_rounds=5)

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt', num_iteration=gbm.best_iteration)

    print('Starting predicting...')
    #prediction
    gbm = lgb.Booster(model_file='model.txt')
    test_text = "XXX 직장인상품이 완화되어행여 바쁘신업무에 지장되시지않도록 안내 남겨드립니다.유선상담을 통해 한도결과 바로 안내드리며본 문자는 기존 XXX 직장인대상XXX고객님을 위한 안내입니다.XXX"
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
    word2ind = word2ind.reshape(1, 73200)
    print(word2ind.shape)

    ypred=gbm.predict(word2ind, num_iteration=gbm.best_iteration)
    print(ypred)
    ypred=np.round(ypred)
    print(ypred)
    ypred=ypred.astype(int)
    print(test_text)
    if ypred==1:
        print("스미싱 문자")
    else:
        print("스미싱 문자 아님")
