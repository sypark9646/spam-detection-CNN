from keras.utils import to_categorical
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import pandas as pd 

import os
from fasttext_dataset import LoadDataset


if __name__ == '__main__':
    MODEL_PATH = "./models/smishing_detect_fasttext"
    data_path = "C:/Users/soyeon/Downloads/training_sample.csv"
    #fasttext.download_model('ko', if_exists='ignore')  # Korean
 
    # Data setup.
    dataset = LoadDataset(csv_path=data_path)

    MAX_WORDS_IN_SEQ = dataset.MAX_WORDS_IN_SEQ
    EMBED_DIM = 300

    data, labels = dataset.DATA_AND_LABEL()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    # Building the model
    input_seq = Input(shape=(MAX_WORDS_IN_SEQ, EMBED_DIM))#, dtype='int32')
    print(input_seq.shape)
    #embed_seq = Embedding(3288, EMBED_DIM, embeddings_initializer='glorot_normal', input_length=MAX_WORDS_IN_SEQ)(input_seq)
    #print(embed_seq.shape)

    #Build model
    conv_1 = Conv1D(128, 5, activation='relu')(input_seq)
    print("conv_1", conv_1.shape)
    conv_1 = MaxPooling1D(pool_size=5)(conv_1)
    print("conv_1_maxpooling1D", conv_1.shape)
    conv_2 = Conv1D(128, 5, activation='relu')(conv_1)
    print("conv_2", conv_2.shape)
    conv_2 = MaxPooling1D(pool_size=5)(conv_2)
    print("conv_2_maxpooling1D", conv_2.shape)
    conv_3 = Conv1D(128, 5, activation='relu')(conv_2)
    print("conv_3", conv_3.shape)
    conv_3 = MaxPooling1D(pool_size=3)(conv_3) #
    print("conv_3_maxpooling1D", conv_3.shape)
    flat = Flatten()(conv_3)
    print("flat", flat.shape)
    flat = Dropout(0.25)(flat)
    print("flat_dropout", flat.shape)
    fc1 = Dense(128, activation='relu')(flat)
    print("dense", fc1.shape)
    dense_1 = Dropout(0.25)(flat)
    print("dense_dropout", dense_1.shape)
    fc2 = Dense(2, activation='softmax')(fc1)
    print("dense", fc2.shape)

    model = Model(input_seq, fc2)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    # Train the model
    model.fit(
        x_train,
        y_train,
        batch_size=20,
        epochs=2,
        callbacks=[ModelCheckpoint(MODEL_PATH, save_best_only=True)],
        validation_data=(x_test, y_test)
    )

    #Save the model
    model.save(MODEL_PATH)

    #Load the model
    model = load_model(MODEL_PATH)
    model.fit( #Fitting
        x_train,
        y_train,
        batch_size=128,
        epochs=5,
        callbacks=[ModelCheckpoint(MODEL_PATH, save_best_only=True)],
        validation_data=[x_test, y_test]
    )

    #Save the model
    model.save(MODEL_PATH)
