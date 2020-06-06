#tokenize using clamp

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
import xml.etree.ElementTree as ET
import subprocess


#collect all noisy words
noisy_words = []
with open('./Data/Extras//Hitachi_Deliverable/MedicalTermAnnotator/ExtraneousTerms', 'r') as file:
    temp = file.read()
    temp = temp.split('\n')[:-1] #ignore blank line at the end
    temp = [word.strip().lower() for word in temp]
noisy_words.extend(temp)
with open('./Data/Extras/Hitachi_Deliverable/MedicalTermAnnotator/ExtraneousSyllables', 'r') as file:
    temp = file.read()
    temp = temp.split('\n')[:-1] #ignore blank line at the end
    temp = [word.strip().lower() for word in temp]
noisy_words.extend(temp)
noisy_words = list(set(noisy_words)-{''})
noisy_words.sort(key=lambda x : len(x), reverse=True)
noisy_words_regex = '|'.join(noisy_words)

def remove_noise(sentence):
    sentence = re.sub(r'(\[\*\*.*\*\*\])|(\d+\:\d+(\:\d+)*(?i)\s*(am|pm)*)|(\d+\/\d+\/\d+)|(\d)|(\|)', '', sentence)
#     sentence = re.sub(fr'\b({noisy_words_regex})\b', ' ', sentence)
    return sentence

filepaths = glob.glob('./Data/downstream_datasets/patient_notes/discharge_summaries/*')

discharge_summaries_tokenized_clamp = dict()
discharge_summaries_sentence_boundaries = dict()
for filepath in tqdm(filepaths):
    filename = filepath.split('/')[-1]
    with open(filepath, 'r') as file:
        text = file.read()
    text = text.lower()
    filename_wo_txt = filename.split('.')[0]
    if(not os.path.exists(f'./Data/downstream_datasets/patient_notes/clamp_output/{filename_wo_txt}.xmi')):
        # print(filename_wo_txt)
        continue
    parser = ET.iterparse(f'./Data/downstream_datasets/patient_notes/clamp_output/{filename_wo_txt}.xmi')
    prev_tag = 'NULL'
    prev_section = 'NULL'
    section_boundary = dict()
    sentence_boundaries = []
    token_boundaries = []
    section_ind = -1
    for event, element in parser:
        tag = element.tag
        tag = re.sub('\{.*\}', '', tag)
        if(tag=='Sentence'):
            section = element.get('segmentId')
            if(prev_section != section):
                section_ind += 1
                section_boundary[section] = dict()
                section_boundary[section]['begin'] = int(element.get('begin'))
                section_boundary[section]['freq'] = 1
            else:
                section_boundary[section]['end'] = int(element.get('end'))
                section_boundary[section]['freq'] += 1
            sentence_boundaries.append([int(element.get('begin')), int(element.get('end'))])
            prev_section = section
        if(tag=='BaseToken'):
            token_boundaries.append([int(element.get('begin')), int(element.get('end'))])
    j = 0
    sentences = []
    sentence = []
    for i in range(len(token_boundaries)):
        if(token_boundaries[i][0]>sentence_boundaries[j][1]):
            j+=1
            if(sentence!=[]):
                sentence = ' '.join(sentence)
                sentence = remove_noise(sentence).strip()
                sentence = re.sub(' +', ' ', sentence)
                sentences.append(sentence.split(' '))
            sentence = []
    #     if(token_boundaries[i][0]>section_boundary[0]['end']):
        token = text[token_boundaries[i][0]:token_boundaries[i][1]].strip()
        token = re.sub(' +', '-', token)
        if(len(token) == 1):
            token = ''
        sentence.append(token)
    discharge_summaries_tokenized_clamp[filename] = sentences
    discharge_summaries_sentence_boundaries[filename] = sentence_boundaries

with open('./Data/downstream_datasets/patient_notes/discharge_summaries_tokenized_clamp.pkl', 'wb') as handle:
    pickle.dump(discharge_summaries_tokenized_clamp, handle)

with open('./Data/downstream_datasets/patient_notes/sentence_boundaries.pkl', 'wb') as handle:
    pickle.dump(discharge_summaries_sentence_boundaries, handle)

