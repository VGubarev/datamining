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

def compute_tfidf(corpus):
    def compute_tf(text):
        tf_text = collections.Counter(text)
        for i in tf_text:
            tf_text[i] = tf_text[i]/float(len(text))
        return tf_text

    def compute_idf(word, corpus):
        return math.log10(len(corpus)/sum([1.0 for i in corpus if word in i]))

    documents_list = []

    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)

        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)

        documents_list.append(tf_idf_dictionary)

    return documents_list


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

# print (corpus)
tfidfs = compute_tfidf(corpus)

for tfidf in tfidfs:
    print (sorted(tfidf.items(), key=operator.itemgetter(1), reverse=True))