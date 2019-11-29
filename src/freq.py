#!/usr/bin/python

import math
import operator
import os
import sys
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 

import collections

printable = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "))

if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
    print ("./tokenizer.py <directory>")
    sys.exit()

texts_tokens = []
tf_texts = []

os.chdir(sys.argv[1])

corpus = []

for file in os.listdir('.'):
    if not os.path.isfile(file):
        continue

    f=open(file, "r")

    text=filter(lambda x: x in printable, f.read())

    text = text.lower()

    tokens = word_tokenize(text)
    texts_tokens.append(tokens)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words] 
    filtered_tokens = [w for w in filtered_tokens if len(w) > 2]
    filtered_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]

    corpus.append(filtered_tokens)

map = {}

for document in corpus:
    for word in document:
        if word not in map.keys():
            map[word] = 1
        else:
            map[word] = map[word] + 1

print (sorted(map.items(), key=operator.itemgetter(1), reverse=True))