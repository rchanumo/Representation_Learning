import numpy as np
import pandas as pd
import pickle
import re
import string


import torch
import torchtext
import torchtext.data as data
from functools import partial
import os
import glob
from progiter import ProgIter as tqdm
import random
import time
import dill
from collections import Counter, defaultdict
import pickle
import fasttext
import itertools


import random
random.seed(32)
# %%pixie_debugger

alt_data_dir = 'Data'

def create_vocab_obj(tokenized_ins, min_treshold=10):
    """
    Find all unique words in corpus (already tokenized) 
    Remove words with fequence of occurrance less than min treshold
    create a torchtext vocab object
    """
    counter = Counter()
    for tokenized_in in tokenized_ins:
        for filename in tokenized_in.keys():
            for sentence in tokenized_in[filename]:
                counter.update(sentence)

    counter = {x : counter[x] for x in counter if counter[x] >= min_threshold}
    counter = Counter(counter) 
    vocab = list(set(counter))

    model = fasttext.load_model('./Data/word_models/BioWordVec_PubMed_MIMICIII_d200.bin')

    embeddings = [model.get_word_vector(word) for word in vocab]
    embeddings = np.stack(embeddings)
    with open('./Data/clamp_related/word_embeddings.pkl', 'wb') as handle:
        pickle.dump(embeddings, handle)

    stoi = {word:ind+2 for ind,word in enumerate(vocab)}
    stoi[' UNK '] = 0
    stoi['<pad>'] = 1
    embeddings = np.concatenate([np.zeros((2,embeddings.shape[1])), embeddings], axis=0)
    embeddings = torch.from_numpy(embeddings).float()

    vocab_obj = torchtext.vocab.Vocab(counter)
    vocab_obj.set_vectors(stoi, embeddings, 200)
    with open('./Data/clamp_related/vocab_obj_ext.pkl', 'wb') as handle:
        pickle.dump(vocab_obj, handle)

class dataiter():

    def __init__(self):
        random.seed(42)
        self.random_state = random.getstate()
        self.batch_size = 4096

    def batch_size_single_fn(self, x, count, sofar):
        """
        Limit total number of documents in batch to 64
        Limit total number of sentences in batch to self.batch_size (=4096)
        """
        if(count > 64):
            return self.batch_size
        return sofar + len(x.ds)

    def get_iters(self):
         """
        Split data into train test and return corresponding iterators.
        """
        TEXT = data.ReversibleField()
        nestedfield = data.NestedField(nesting_field=TEXT,
                                      include_lengths=True,)
        LABEL = data.ReversibleField(sequential=False, use_vocab=False, is_target=True)
        FILENAME = data.ReversibleField(sequential=False)

        #create train iter
        print('\n preparing train data')
        discharge_summaries_annotated_tokenized_dict = pickle.load(open(f'./{alt_data_dir}/discharge_summaries_tokenized_clamp.pkl', 'rb'))
        discharge_summaries_tokenized_dict = pickle.load(open(f'./{alt_data_dir}/discharge_summaries_tokenized_clamp.pkl', 'rb'))
        discharge_summaries_tokenized_dict_obesity = pickle.load(open(f'./{alt_data_dir}/downstream_datasets/Obesity/discharge_summaries_tokenized_clamp.pkl', 'rb'))
        discharge_summaries_tokenized_dict_smoking = pickle.load(open(f'./{alt_data_dir}/downstream_datasets/Smoking/discharge_summaries_tokenized_clamp.pkl', 'rb'))
        discharge_summaries_tokenized_dict_similarity = pickle.load(open(f'./{alt_data_dir}/downstream_datasets/Similarity/discharge_summaries_tokenized_clamp.pkl', 'rb'))
        discharge_summaries_tokenized_dict_patient_notes = pickle.load(open(f'./{alt_data_dir}/downstream_datasets/patient_notes/discharge_summaries_tokenized_clamp.pkl', 'rb'))

        
        if(not os.path.exists('./Data/clamp_related/vocab_obj_ext.pkl')):
            create_vocab_obj([discharge_summaries_tokenized_dict,
                            discharge_summaries_tokenized_dict_obesity,
                            discharge_summaries_tokenized_dict_smoking,
                            discharge_summaries_tokenized_dict_similarity,
                            discharge_summaries_tokenized_dict_patient_notes])
            print("Created vocab obj: rerun")
            assert 1==2

        TEXT.vocab = pickle.load(open('./Data/clamp_related/vocab_obj_ext.pkl', 'rb'))

        counter = Counter()
        counter.update(list(discharge_summaries_tokenized_dict.keys())
            +list(discharge_summaries_tokenized_dict_obesity.keys())
            +list(discharge_summaries_tokenized_dict_smoking.keys())
            +list(discharge_summaries_tokenized_dict_similarity.keys())
            +list(discharge_summaries_tokenized_dict_patient_notes.keys()))
        FILENAME.vocab = torchtext.vocab.Vocab(counter)

        #fromm
        diagnoses_label_dict = pickle.load(open('./Data/diagnoses_files/diagnoses_dict_ctakes_ext.pkl', 'rb'))
        procedure_label_dict = pickle.load(open('./Data/procedure_files/procedure_dict_ctakes_ext.pkl', 'rb'))
        medicine_label_dict = pickle.load(open('./Data/medicine_files/medicine_dict_ctakes_ext.pkl', 'rb'))

        filenames = list(discharge_summaries_tokenized_dict.keys())+\
                    list(discharge_summaries_tokenized_dict_obesity.keys())+\
                    list(discharge_summaries_tokenized_dict_smoking.keys())+\
                    list(discharge_summaries_tokenized_dict_similarity.keys())+\
                    list(discharge_summaries_tokenized_dict_patient_notes.keys())

        discharge_summaries_tokenized_dict.update(discharge_summaries_tokenized_dict_obesity)
        discharge_summaries_tokenized_dict.update(discharge_summaries_tokenized_dict_smoking)
        discharge_summaries_tokenized_dict.update(discharge_summaries_tokenized_dict_similarity)
        discharge_summaries_tokenized_dict.update(discharge_summaries_tokenized_dict_patient_notes)
        
        examples = []
        for filename in filenames:
            file = discharge_summaries_tokenized_dict[filename]
            # file = [sentence for sentence in file if(sentence!=[] and sentence!=[''])]
            file_len = 0
            file_proc = []
            for sentence in file:
                if(sentence!=[] and sentence!=['']):
                    file_proc.append(sentence)
                    file_len += len(sentence)
                    if(file_len > 25000):
                        break
            file = file_proc
            if(len(file)<2 or diagnoses_label_dict.get(filename) is None or procedure_label_dict.get(filename) is None or medicine_label_dict.get(filename) is None):
                continue
            example = data.Example()
            example = example.fromlist([filename, file],
                                        [('filename', FILENAME), ('ds', nestedfield)])
            examples.append(example)
        print('Size of data: ', len(examples), '\n')
        full_dataset = data.Dataset(examples, [('ds', nestedfield), ('filename', FILENAME)])
        train_dataset, test_dataset = full_dataset.split(split_ratio=[0.8, 0.2], random_state=random.getstate())

        train_iter = data.Iterator(full_dataset, #change to train_dataset with proper stratification
                                          batch_size = self.batch_size,
                                        batch_size_fn = self.batch_size_single_fn,
                                          sort_key = lambda x: len(x.ds),
                                          repeat = False,
                                          shuffle = True,
                                          sort_within_batch = True)
        test_iter = data.BucketIterator(test_dataset,
                                          batch_size = self.batch_size,
                                        batch_size_fn = self.batch_size_single_fn,
                                          sort_key = lambda x: len(x.ds),
                                          repeat = False,
                                          shuffle = False,
                                          sort_within_batch = True)


        examples = []
        for filename in discharge_summaries_annotated_tokenized_dict.keys():
            file = discharge_summaries_annotated_tokenized_dict[filename]
            if(len(file)<2):
                continue
            example = data.Example()
            example = example.fromlist([filename, file],
                                        [('filename', FILENAME), ('ds', nestedfield)])
            examples.append(example)

        singular_dataset = data.Dataset(examples, [('filename', FILENAME), ('ds', nestedfield)])
        singular_iter = data.Iterator(singular_dataset,
                                          batch_size = 1,
                                          repeat = False,
                                          shuffle = True)


        return train_iter, test_iter, singular_iter, TEXT, FILENAME
