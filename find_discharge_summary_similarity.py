import numpy as np
import pandas as pd
from collections import defaultdict
import os
import time
import sys
import glob
import re
import random
from math import ceil
# from tqdm import tqdm_notebook as tqdm
from progiter import ProgIter as tqdm
import regex
import pickle
import itertools
import matplotlib.pyplot as plt
from sklearn.utils import resample
import multiprocessing as mp
from scipy.sparse import csr_matrix
from itertools import islice
import gc

'''
Find similarity between discharge summaries using:
 -diseases
 -Medicines
 -Procedures
'''
def jaccard(list1, list2):
    a = set(list1)
    b = set(list2)
    c = a.intersection(b)
    if(not len(c)):
        return 0.0
    return float(len(c)) / (len(a) + len(b) - len(c))

def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

#prepare dictionary semantic type ->  disease(d)/procedure(p)/medicine(m)
semanticType2category = {'acab':'d', 'anab':'d', 'bact':'d', 'cgab':'d', 'comd':'d', 'dsyn':'d',
                        'emod':'d', 'mobd':'d', 'sosy':'d', 'virs':'d', 'antb':'m', 'chem':'m',
                        'clnd':'m', 'enzy':'m', 'orch':'m', 'phsu':'m', 'strd':'m', 'topp':'p',
                        'irda':'p', 'lbpr':'p', 'diap':'p'}

annotations_dict = dict() # divide required medical entities into groups
filepaths = glob.glob('./Data/discharge_summaries_annotations/noteevents*')
for filepath in tqdm(filepaths):
    filename = filepath.split('/')[-1]
    with open(filepath, 'r') as file:
        annotations_dict[filename] = {'m':[], 'd':[], 'p':[]} # create container to accumulate medical entities corresponding to medicine, procedure and disease
        annotations = file.read()
        annotations = annotations.split('\n')[:-1] #ignore blank line at the end
        medical_entities = []
        semantic_types = []
        for annotation in annotations:
            annotation = annotation.split('|')
            medical_entities.append(annotation[4].lower())
            semantic_types.append(annotation[7])
    for i in range(len(medical_entities)):
        if(semantic_types[i] in semanticType2category.keys()):
            annotations_dict[filename][semanticType2category[semantic_types[i]]].append(medical_entities[i])

fileencode = {filename:i for i,filename in enumerate(annotations_dict.keys())}
filedecode = {fileencode[filename]:filename for filename in annotations_dict.keys()}
start = time.time()
filenames = list(annotations_dict.keys())
# for file1name, file2name in tqdm(itertools.product(filenames, filenames)):
def find_pairwise_similarity(filepair):
    file1name = filepair[0]
    file2name = filepair[1]
    if(file1name == file2name):
        return None
    medicine_similarity = jaccard(annotations_dict[file1name]['m'], annotations_dict[file2name]['m'])
    disease_similarity = jaccard(annotations_dict[file1name]['d'], annotations_dict[file2name]['d'])
    procedure_similarity = jaccard(annotations_dict[file1name]['p'], annotations_dict[file2name]['p'])
#     print(medicine_similarity, disease_similarity, procedure_similarity)
    score = max([medicine_similarity, disease_similarity, procedure_similarity])
#     if(max([medicine_similarity, disease_similarity, procedure_similarity]) == 1):
#         score_norm = 0.8
    if(score > 0.8):
        score_norm = 0.8
    elif(score > 0.6):
        score_norm = 0.6
    elif(score > 0.4):
        score_norm = 0.4
    elif(score > 0.2):
        score_norm = 0.2
    else:
        score_norm = 0
    if(score_norm == 0):
        return None
    if((score_norm==0.2 or score_norm==0.4) and random.choice(list(range(100)))!=0):
        return None
    return((fileencode[file1name],fileencode[file2name], score_norm))

cores = mp.cpu_count()
output = []
iterator_batch_size = 7500*14999
with mp.Pool(processes=cores) as pool:
#     output = list(pool.map(find_pairwise_similarity, itertools.combinations(filenames, 2)))
    for iterator_batch in tqdm(split_every(iterator_batch_size, itertools.combinations(filenames, 2))):
        output.extend(list(pool.map(find_pairwise_similarity, iterator_batch)))
        gc.collect()
rows = [elem[0] for elem in output if elem is not None]
cols = [elem[1] for elem in output if elem is not None]
data = [elem[2] for elem in output if elem is not None]
matrix = csr_matrix((data, (rows, cols)), shape=(len(filenames), len(filenames)))
print(time.time() - start)
with open('./Data/similarity_matrix.pkl', 'wb') as handle:
    pickle.dump(matrix, handle)
with open('./Data/fileencode.pkl', 'wb') as handle:
    pickle.dump(fileencode, handle)

#score=0 include dissimilar documents as well as  upper traingle in matrix
filepairs_ind_sampled = np.argwhere(matrix==0.8)
n_samples_p = len(filepairs_ind_sampled)
for i in [0.2, 0.4, 0.6]:
    temp = np.argwhere(matrix==i)
    temp_idx = np.random.randint(temp.shape[0], size=n_samples_p)
    temp = temp[temp_idx]
    filepairs_ind_sampled = np.concatenate([filepairs_ind_sampled, temp], axis=0)
filepairs_ind_sampled = filepairs_ind_sampled.tolist()
filepairs_sampled = [(filedecode[file1ind], filedecode[file2ind]) for (file1ind, file2ind) in filepairs_ind_sampled]
labels = np.zeros((len(filepairs_sampled),), dtype=int)
labels[:int(n_samples_p)] = 1
pairwise_data = (filepairs_sampled, labels)
with open('./Data/pairwise_data.pkl', 'wb') as handle:
    pickle.dump(pairwise_data, handle)
