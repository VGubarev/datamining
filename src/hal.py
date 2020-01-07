#!/usr/bin/python

import csv
import collections
import math
import operator
import os
import sys
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from scipy.sparse import lil_matrix

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()


printable = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "))

hal = {}


def get_lemmatized_tokens(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    filtered_tokens = [w for w in filtered_tokens if len(w) > 2]
    filtered_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return filtered_tokens


def fill_hal(tokens, window_size):
    tokens_len = len(tokens)

    for i in range(tokens_len):
        end_i = tokens_len
        if i + window_size + 1 < tokens_len:
            end_i = i + window_size + 1
        key = tokens[i]
        for wi in range(i + 1, end_i, 1):
            if i == wi:
                continue
            weight = window_size - abs(i - wi) + 1
            if tokens[wi] in hal[key]:
                hal[key][tokens[wi]] += weight
            else:
                hal[key][tokens[wi]] = weight


def build_sparse_matrix(hal, indexes):
    # construct an empty matrix with shape (M, N)
    hal_size = len(hal)
    mat = lil_matrix((hal_size, hal_size), dtype=float)
    for row_token, relations in hal.items():
        for column_token, weight in relations.items():
            mat[indexes[column_token], indexes[row_token]] = weight
    return mat

def compute_scalar(sparse_matrix, indexes):
    result = {}
    for word, index in indexes.items():
        for word2, index2 in indexes.items():
            if (word == word2) or ((word, word2) in result):
                continue
            scalar = sparse_matrix[index2, index]
            result[(word2, word)] = scalar
    
    return sorted(result, key=operator.itemgetter(1), reverse=True)

vector_cache = {}

def get_vector(word, index):
    if word in vector_cache:
        return vector_cache[word]

    column = sparse_matrix[:, index].toarray()
    row = sparse_matrix[index, :].toarray()
    row.shape = (row.shape[1], row.shape[0])
    vector = np.row_stack((column, row))
    vector_cache[word] = vector
    return vector

def compute_similarity(sparse_matrix, indexes):
    result = {}
    for word, index in indexes.items():
        lhs = get_vector(word, index)
        for word2, index2 in indexes.items():
            if (word == word2) or ((word2, word) in result):
                continue
            rhs = get_vector(word2, index2)
            similarity = cosine_similarity(np.transpose(lhs), np.transpose(rhs), dense_output=False)
            result[(word, word2)] = similarity[0][0]

        if index % 100 == 0:
            print("\tFinished to process \"" + word + "\"")
    
    return sorted(result.items(), key=operator.itemgetter(1), reverse=True)

if len(sys.argv) < 4 or not os.path.isfile(sys.argv[1]) or not os.path.isdir(sys.argv[3]):
    print ("./hal.py <filename> <integer distance> <result base dir>")
    sys.exit()

try:
    int(sys.argv[2])
except ValueError:
    print ("./hal.py <filename> <integer distance> <result base dir>")
    sys.exit()

filename = sys.argv[1]
window_size = sys.argv[2]
res_base_dir = sys.argv[3]

with open(filename) as text:
    text = text.read()

text = filter(lambda x: x in printable, text)
text = text.lower()

print("Lemmatize file")

tokens = get_lemmatized_tokens(text)

indexes = {}

for token in tokens:
    hal[token] = {}
    if token not in indexes:
        indexes[token] = len(indexes)

print("Fill HAL matrix")
fill_hal(tokens, int(window_size))
sparse_matrix = build_sparse_matrix(hal, indexes)

print("Save sparse HAL matrix on a disk")
dir_name = res_base_dir + "/hal" + str(window_size)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

with open(dir_name + "/" + os.path.basename(filename) + ".csv", "w") as smf:
    writer = csv.writer(smf, delimiter=';')
    sorted_indexes = sorted(indexes.items(), key=operator.itemgetter(1))
    head = [""]
    rows = []
    for pair in sorted_indexes:
        head.append(str(pair[0]))
        row = sparse_matrix[pair[1], :].toarray().ravel().tolist()
        row.insert(0, str(pair[0]))
        rows.append(row)
    writer.writerow(head)
    writer.writerows(rows)

print("Compute scalar")
result = compute_scalar(sparse_matrix, indexes)
with open(dir_name + "/" + os.path.basename(filename) + ".scalar", "w") as f:
    f.write(str(result))
print(result[:30])

print("Compute similarity")
result = compute_similarity(sparse_matrix, indexes)
with open(dir_name + "/" + os.path.basename(filename) + ".similarity", "w") as f:
    f.write(str(result))
print(result[:30])