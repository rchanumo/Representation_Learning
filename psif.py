import fasttext

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import os
import time
import sys
import glob
import re
import random
from math import ceil
from progiter import ProgIter as tqdm
import regex
import pickle
import time
import tables


'''
Implementation:
Unsupervised Document Representation using Partition Word-Vectors Averaging
    Vivek Gupta, Ankit Kumar Saw, Partha Pratim Talukdar, Praneeth Netrapalli

'''
import numpy as np
from sklearn.decomposition import DictionaryLearning, SparseCoder, MiniBatchDictionaryLearning
from sklearn.decomposition import PCA

#preprocessing steps

#find vocabulary(V) from input
V = pickle.load(open('./Data/vocab.pkl', 'rb'))
word2idx = {word:i for i,word in enumerate(V)}
idx2word = {i:word for i,word in enumerate(V)}

#find word embeddings
word_embeddings = pickle.load(open('./Data/word_embeddings.pkl', 'rb'))

#find word relative frequency
word_freqency = Counter()
discharge_summaries_dict = pickle.load(open('./Data/discharge_summaries_tokenized_dict.pkl', 'rb'))
for filename in discharge_summaries_dict.keys():
    for sentence in discharge_summaries_dict[filename]:
        word_freqency.update(sentence)
word_freqency = {word:word_freqency[word] for word in V}
# with open('./Data/vocab_with_word_frequency.pkl', 'wb') as handle:
#     pickle.dump(word_freqency, handle)
n_words = sum(list(word_freqency.values()))
pw = dict()
for word in word_freqency.keys():
#     if(word in V):
    try:
        pw[word2idx[word]] = word_freqency[word]/n_words
    except:
        print(word)

# Dictionary learning for word-vector
n_components = 60
dl = MiniBatchDictionaryLearning(n_components=n_components, n_jobs=-2, verbose=True)
dl.fit(word_embeddings)
print('Dictionary fitted')
dictionary_atoms = dl.components_
sparse_coder = SparseCoder(dictionary_atoms, n_jobs=-1)
sparse_coeffs = sparse_coder.transform(word_embeddings)
print('Completed Dictionary learning')
with open('./Data/sparse_coeffs.pkl', 'wb') as handle:
    pickle.dump(sparse_coeffs, handle)
# sparse_coeffs = pickle.load(open('./Data/sparse_coeffs.pkl', 'rb'))

# use pytables to store word vectors
n, d  = word_embeddings.shape[0], word_embeddings.shape[1]
fileh = tables.open_file('topic_vectors.h5', mode='w')
atom = tables.Float32Atom()
filters = tables.Filters(complevel=1, complib='blosc', fletcher32=True)
topic_vectors = fileh.create_carray(fileh.root, 'data', atom, (n,n_components*d), filters=filters)

#Word topic-vector formation
chunk_size = 10000
for chunk_ind in range(0,n,chunk_size):
    topic_vectors_chunk = []
    for i in range(n_components):
        weighted_embeddings_chunk = word_embeddings[chunk_ind:chunk_ind+chunk_size]*(np.expand_dims(sparse_coeffs[chunk_ind:chunk_ind+chunk_size,i], axis=1))
        topic_vectors_chunk.append(weighted_embeddings_chunk)
    topic_vectors_chunk = np.concatenate(topic_vectors_chunk, axis=1)
    topic_vectors[chunk_ind:min(n, chunk_ind+chunk_size),:] = topic_vectors_chunk

fileh = tables.open_file('./Data/psif/topic_vectors.h5', 'r')
topic_vectors = fileh.root.data

# #SIF reweighed document vector embedding
# document_embeddings = dict()
# for filename in tqdm(discharge_summaries_dict.keys()):
#     try:
#         document = discharge_summaries_dict[filename]['history of present illness']
#     except:
#         try:
#             document = discharge_summaries_dict[filename]['hpi']
#         except:
#             print(filename)
#             print(discharge_summaries_dict[filename].keys())
#             continue
#     document = [word2idx[word] for word in document if word in V]
#     document_embedding = []
#     for word_idx in document:
#         weighted_topic_vector = (0.001/(0.001+pw[word_idx])) * topic_vectors[word_idx,:]
#         document_embedding.append(weighted_topic_vector)
#     document_embedding = sum(document_embedding)/len(document_embedding)
#     document_embeddings[filename] = document_embedding

# #remove first principal component from embedding
# X = list(document_embeddings.values()) #Collection of document embeddings
# X = np.stack(X)
# pca = PCA(n_components=2)
# pca.fit(X)
# first_singular_vector = pca.components_[0,:]

# for filename in document_embeddings.keys():
#     document_embeddings[filename] -= (np.linalg.norm(first_singular_vector,2)**2) * document_embeddings[filename] #remove first principal component
