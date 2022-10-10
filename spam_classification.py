#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from collections import Counter
import re, string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
import numpy as np


    

def def_vocab(data, vocab):
    re_punt = re.compile(f"[{re.escape(string.punctuation)}]")
    stop_words = set(stopwords.words("english"))    
    for line in data:
        tokens = line.split()
        tokens = [re_punt.sub('', w) for w in tokens]
        tokens = [w for w in tokens if w.isalpha()]
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [w.lower() for w in tokens]
        vocab.update(tokens)
        
        
def clean_data(data, vocab):
    ret_data = list()
    for line in data:
        tokens = line.split()
        tokens = [w for w in tokens if w in vocab.keys()]
        ret_data.append(' '.join(tokens))    
    return ret_data

# def prepare_data(data_file):    
#     data = pd.read_csv(data_file) 
  
#     target = list()
#     lines = list()    
#     for i in data['is_depression']:
#         target.append(i)    
#     for i in data['clean_text']:
#         lines.append(i)
#     return lines, target
 

def create_tokenizer(lines):
    t = Tokenizer()
    t.fit_on_texts(lines)
    return t

def encode_data(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen = max_length, padding='post')
    return padded
 
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    # model.add(Dropout(0.5))
    model.add(Conv1D(32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model
      



  
data = pd.read_csv('./spam.csv')
X = data['MESSAGE']
y = data['CATEGORY']
  
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, random_state=10,
                                                test_size=0.15)
vocab = Counter()
def_vocab(Xtrain, vocab)

Xtrain = clean_data(Xtrain, vocab)
Xtest = clean_data(Xtest, vocab)

t = create_tokenizer(Xtrain)

vocab_size = len(t.word_index) + 1
print(f"vocab length: {vocab_size}")

max_length = max([len(d.split()) for d in Xtrain])
print(f"Max length for document: {max_length}")

Xtrain = encode_data(t, max_length, Xtrain)
Xtest = encode_data(t, max_length, Xtest)

# ytrain = to_categorical(ytrain, num_classes=4)
# ytest = to_categorical(ytest, num_classes=4)

model = define_model(vocab_size, max_length)
model.fit(Xtrain, np.array(ytrain), epochs=100)
model.save('model.h5')



# _, acc = model.evaluate(Xtest, ytest)
# print(f"Accuracy on test: {acc*100}")

