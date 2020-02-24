# coding: utf-8
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from fasttext_dataset import LoadDataset

#this code is from [Python Machine Learning Cookbook]

if __name__ == '__main__':
    data_path = "C:/Users/soyeon/Downloads/training_sample.csv"

    print('Loading data...')
    # Data setup.
    dataset = LoadDataset(csv_path=data_path)
    data, labels = dataset.DATA_AND_LABEL()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    #to reduce dimensionality
    x_train = x_train.reshape(160, 73200)
    x_test = x_test.reshape(40, 73200)
    y_trains = [a for a, b in y_train]
    y_tests = [a for a, b in y_test]
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
        'learning_rate':0.05,
        'feature_fraction':0.9,
        'bagging_fraction':0.8,
        'bagging_freq':5,
        'verbose':0
    }

    print('Starting training...')
    # train
    gbm=lgb.train(parameters, lgbTrain, num_boost_round=20, valid_sets=lgbTrain)

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Starting predicting...')
    #prediction
    ypred=gbm.predict(x_test, num_iteration=gbm.best_iteration)
    ypred=np.round(ypred)
    ypred=ypred.astype(int)
    print('rmse of the model is:', mean_squared_error(y_test, ypred)**0.5)
    #to analyze the errors that were made in the binary classification in more detail,
    #we need to compute the confusion matrix
    confmatrix=confusion_matrix(y_test,ypred)
    print(confmatrix)
    print(accuracy_score(y_test, ypred))
