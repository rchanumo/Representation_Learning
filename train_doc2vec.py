'''
train doc2vec model
'''

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
import regex
import pickle
import time
from progiter import ProgIter as tqdm


discharge_summaries_tokenized = pickle.load(open('./Data/discharge_summaries_tokenized_clamp.pkl', 'rb'))
discharge_summaries_tokenized_smoking = pickle.load(open('./Data/downstream_datasets/Smoking/discharge_summaries_tokenized_clamp.pkl', 'rb'))
discharge_summaries_tokenized_similarity = pickle.load(open('./Data/downstream_datasets/Similarity/discharge_summaries_tokenized_clamp.pkl', 'rb'))
discharge_summaries_tokenized_obesity = pickle.load(open('./Data/downstream_datasets/Obesity/discharge_summaries_tokenized_clamp.pkl', 'rb'))
vocab = pickle.load(open('./Data/vocab.pkl', 'rb'))

docs = []
for tokenized in [discharge_summaries_tokenized, discharge_summaries_tokenized_smoking, discharge_summaries_tokenized_similarity, discharge_summaries_tokenized_obesity]:
    for filename in tqdm(tokenized.keys()):
        discharge_summary = []
        for sentence in tokenized[filename]:
            discharge_summary.extend(sentence)
        discharge_summary = ' '.join(discharge_summary)
        discharge_summary = gensim.utils.to_unicode(discharge_summary).split() # review utf-8 handling
        docs.append(TaggedDocument(discharge_summary, [filename]))


#model meta
save_path = './Data/doc2vec/doc2vec_1.bin'
algorithm = 'pv_dbow'
window = 15
vector_size = 256
alpha = 0.025
min_alpha = 0.001
num_epochs = 100
vocab_min_count = 1
sampling_threshold = 1e-5
cores = 4
negative = 5
hs = 0

vocab = pickle.load(open('./Data/vocab.pkl', 'rb'))
word_vectors = pickle.load(open('./Data/word_embeddings.pkl', 'rb'))
# Save to file
with open('./Data/pretrained_embeddings_for_doc2vec.txt', 'w', encoding='utf-8') as fout:
    for i in tqdm(range(len(vocab))):
        fout.write(u"{} {}\n".format(vocab[i], ' '.join(word_vectors[i].astype(str))))
pretrained_emb = "./Data/pretrained_embeddings_for_doc2vec.txt"

if algorithm == 'pv_dmc':
    # PV-DM with concatenation
    # window=5 (both sides) approximates paper's 10-word total window size
    # PV-DM w/ concatenation adds a special null token to the vocabulary: '\x00'
    model = Doc2Vec(dm=1, dm_concat=1, vector_size=vector_size, window=window, negative=negative, hs=hs,
                    min_count=vocab_min_count, pretrained_emb=pretrained_emb, sample=sampling_threshold, workers=cores)
elif algorithm == 'pv_dma':
    # PV-DM with average
    # window=5 (both sides) approximates paper's 10-word total window size
    model = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size, window=window, negative=negative, hs=hs,
                    min_count=vocab_min_count, pretrained_emb=pretrained_emb, sample=sampling_threshold, workers=cores)
elif algorithm == 'pv_dbow':
    # PV-DBOW
    # model = Doc2Vec(dm=0, vector_size=vector_size, window=window, negative=negative, hs=hs,
    #                 min_count=vocab_min_count, pretrained_emb=pretrained_emb, sample=sampling_threshold, workers=cores)
    model = Doc2Vec(dm=0, vector_size=vector_size, window=window, negative=negative, hs=hs,
                    min_count=vocab_min_count, sample=sampling_threshold, workers=cores)

else:
    raise ValueError('Unknown algorithm: %s' % algorithm)


model.build_vocab(docs)
vocab_size = len(model.wv.vocab)

print('Training started')
model.train(docs, total_examples=len(docs), epochs=num_epochs, start_alpha=alpha, end_alpha=min_alpha)
print('Training finished')
model.save(save_path)
