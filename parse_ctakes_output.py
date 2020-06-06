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
import xml.etree.ElementTree as ET
import subprocess

from icd9cms import search as search_d
from icd9pcs import search as search_p

import json

'''
parse ctakes output
'''
filepaths = glob.glob('./Data/ctakes_output/noteevents_*')+\
            glob.glob('./Data/downstream_datasets/Obesity/ctakes_output/*')+\
            glob.glob('./Data/downstream_datasets/Smoking/ctakes_output/*')+\
            glob.glob('./Data/downstream_datasets/patient_notes/ctakes_output/*')+\
            glob.glob('./Data/downstream_datasets/Similarity/ctakes_output/*')

medical_concepts_dict = dict()
for filepath in tqdm(filepaths):
    filename = filepath.split('/')[-1].split('.')[0]
    filename = f'{filename}.txt'
    parser = ET.iterparse(filepath)
    xmi_id_dict = dict()
    for event, element in parser:
        tag = element.tag
        tag = re.sub('\{.*\}', '', tag)
        if(tag == 'MedicationMention'):
    #         print(element.attrib)
            ontology_array = element.get('ontologyConceptArr').split()
            for xmi_id in ontology_array:
                xmi_id_dict[xmi_id] = dict()
                xmi_id_dict[xmi_id]['section'] = 'm'
                xmi_id_dict[xmi_id]['polarity'] = element.get('polarity')
        elif(tag == 'DiseaseDisorderMention' or tag == 'SignSymptomMention'):
            ontology_array = element.get('ontologyConceptArr').split()
            for xmi_id in ontology_array:
                xmi_id_dict[xmi_id] = dict()
                xmi_id_dict[xmi_id]['section'] = 'd'
                xmi_id_dict[xmi_id]['polarity'] = element.get('polarity')
        elif(tag == 'ProcedureMention'):
            ontology_array = element.get('ontologyConceptArr').split()
            for xmi_id in ontology_array:
                xmi_id_dict[xmi_id] = dict()
                xmi_id_dict[xmi_id]['section'] = 'p'
                xmi_id_dict[xmi_id]['polarity'] = element.get('polarity')
    medical_concepts_dict[filename] = xmi_id_dict
with open('./Data/clamp_output/medical_concepts_dict.pkl', 'wb') as handle:
    pickle.dump(medical_concepts_dict, handle)


'''
collect icd codes and medicine names
'''
medical_concepts_dict = pickle.load(open('./Data/clamp_output/medical_concepts_dict.pkl', 'rb'))
filepaths = glob.glob('./Data/downstream_datasets/patient_notes/ctakes_output/*')+\
            glob.glob('./Data/ctakes_output/noteevents_*')+\
            glob.glob('./Data/downstream_datasets/Obesity/ctakes_output/*')+\
            glob.glob('./Data/downstream_datasets/Smoking/ctakes_output/*')+\
            glob.glob('./Data/downstream_datasets/Similarity/ctakes_output/*')
procedure_dict_ctakes = dict()
diagnoses_dict_ctakes = dict()
medicine_dict_ctakes = dict()
for filepath in tqdm(filepaths):
    filename = filepath.split('/')[-1].split('.')[0]
    filename = f'{filename}.txt'
    if(medical_concepts_dict.get(filename) is None):
        continue
    # print(filename)
    parser = ET.iterparse(filepath)
    xmi_id_dict = dict()
    disease_list = set()
    procedure_list = set()
    medicine_list = set()
    for event, element in parser:
        tag = element.tag
        tag = re.sub('\{.*\}', '', tag)
        if(tag == 'UmlsConcept'):
            if(element.get('codingScheme') == 'ICD9CM' or element.get('codingScheme') == 'MTHICD9'):
                xmi_id = element.get('{http://www.omg.org/XMI}id')
    #             print(element.attrib)
                if(medical_concepts_dict[filename].get(xmi_id) is not None):
                    if(medical_concepts_dict[filename][xmi_id]['polarity'] != '-1'):
                        if(medical_concepts_dict[filename][xmi_id]['section'] == 'd'):
                            icd_code = element.get('code')
                            if('.' in icd_code):
                                icd_code = icd_code.split('.')[0]
                            if(search_d(icd_code) is not None):
                                disease_list.add(icd_code)
                        elif (medical_concepts_dict[filename][xmi_id]['section'] == 'p'):
                            icd_code = element.get('code')
                            if('.' in icd_code):
                                icd_code = icd_code.split('.')[0]
                            if(search_p(icd_code) is not None):
                                procedure_list.add(icd_code)
                else:
                    print('none')
            elif(element.get('codingScheme') == 'RXNORM'):
                xmi_id = element.get('{http://www.omg.org/XMI}id')
                if(medical_concepts_dict[filename].get(xmi_id) is not None):
                    if(medical_concepts_dict[filename][xmi_id]['polarity'] != '-1'):
                        medicine = element.get('preferredText')
                        if(medicine is not None):
                            medicine = medicine.lower()
                            medicine_list.add(medicine)

    disease_list = list(disease_list)
    diagnoses_dict_ctakes[filename] = disease_list
    procedure_list = list(procedure_list)
    procedure_dict_ctakes[filename] = procedure_list
    medicine_list = list(medicine_list)
    medicine_dict_ctakes[filename] = medicine_list
with open('./Data/diagnoses_files/diagnoses_dict_ctakes_ext.pkl', 'wb') as handle:
    pickle.dump(diagnoses_dict_ctakes, handle)
with open('./Data/procedure_files/procedure_dict_ctakes_ext.pkl', 'wb') as handle:
    pickle.dump(procedure_dict_ctakes, handle)
with open('./Data/medicine_files/medicine_dict_ctakes_ext.pkl', 'wb') as handle:
    pickle.dump(medicine_dict_ctakes, handle)

'''
process codes
'''
from sklearn.preprocessing import MultiLabelBinarizer

def filter_icd_codes(codes, search):
    #remove parent in list if child exists
    codes_filtered = []
    for icd_code in codes:
        node = search(icd_code)
        descendants = node.descendants()
        descendants = [search(code).alt_code for code in descendants]
        if(len(set(descendants).intersection(codes)) == 0):
            codes_filtered.append(icd_code)
    return codes_filtered

def add_parents(codes, search):
    out_codes = set()
    out_codes.update(codes)
    for code in codes:
        node = search(code)
        out_codes.update([search(ancestor).alt_code for ancestor in node.ancestors()])
    return list(out_codes)

def get_mask(codes, ctakes_out, search):
    out_codes = set()
    for code in codes:
        node = search(code)
        descendants = [search(descendant).alt_code for descendant in node.descendants()]
        out_codes.update(descendants)
    out_codes = list(out_codes)
    out_codes = list(set(out_codes).intersection(ctakes_out))
    return out_codes

def preprocess_medicine_name(medicine_name):
    medicine_name = medicine_name.lower()
    medicine_name = re.sub(r'\(.*\)', '', medicine_name)
    medicine_name = re.sub(r'[^\ a-z]|', '', medicine_name)
    medicine_name = re.sub(r'\band\b', '', medicine_name)
    medicine_name = re.sub(r' +', ' ', medicine_name)
    return medicine_name.strip()

diagnoses_dict_ctakes = pickle.load(open('./Data/diagnoses_files/diagnoses_dict_ctakes_ext.pkl', 'rb'))
procedure_dict_ctakes = pickle.load(open('./Data/procedure_files/procedure_dict_ctakes_ext.pkl', 'rb'))
medicine_dict_ctakes = pickle.load(open('./Data/medicine_files/medicine_dict_ctakes_ext.pkl', 'rb'))


# get diagnoses labels and mask
diagnoses_ctakes_out = set()
for filename in diagnoses_dict_ctakes.keys():
    diagnoses_ctakes_out.update(filter_icd_codes(diagnoses_dict_ctakes[filename], search_d))
# with open('./Data/diagnoses_files/diagnoses_vocab.pkl', 'wb') as handle:
#     pickle.dump(diagnoses_vocab, handle)

diagnoses_label_dict = dict()
diagnoses_mask_dict = dict()
for filename in tqdm(diagnoses_dict_ctakes.keys()):
    diagnoses_list = diagnoses_dict_ctakes[filename]
    diagnoses_list_filtered = filter_icd_codes(diagnoses_list, search_d)
    diagnoses_mask = get_mask(diagnoses_list_filtered, diagnoses_ctakes_out, search_d)
    diagnoses_mask_dict[filename] = diagnoses_mask
    diagnoses_final_list = add_parents(diagnoses_list_filtered, search_d)
    diagnoses_label_dict[filename] = diagnoses_final_list
#one hot encoding
#labels
filenames = list(diagnoses_label_dict.keys())
labels = list(diagnoses_label_dict.values())
d_counter = Counter()
for label_list in labels:
    d_counter.update(label_list)
keep = [x for x in list(set(d_counter)) if d_counter[x]>=50]
mlb = MultiLabelBinarizer()
mlb.fit([keep,])
one_hot_encoded = mlb.transform(labels)
print(len(mlb.classes_))
diagnoses_label_dict = {filename:label for filename,label in zip(filenames, one_hot_encoded)}

#mask
filenames = list(diagnoses_mask_dict.keys())
mask = list(diagnoses_mask_dict.values())
one_hot_encoded = mlb.transform(mask)
one_hot_encoded = 1 - one_hot_encoded
diagnoses_mask_dict = {filename:m for filename,m in zip(filenames, one_hot_encoded)}

with open('./Data/diagnoses_files/diagnoses_label_dict_ext.pkl', 'wb') as handle:
    pickle.dump(diagnoses_label_dict, handle)
with open('./Data/diagnoses_files/diagnoses_mask_dict_ext.pkl', 'wb') as handle:
    pickle.dump(diagnoses_mask_dict, handle)
with open('./Data/diagnoses_files/diagnoses_label_encoder_ext.pkl', 'wb') as handle:
    pickle.dump(mlb, handle)

# get procedure labels and mask
procedure_ctakes_out = set()
for filename in procedure_dict_ctakes.keys():
    procedure_ctakes_out.update(filter_icd_codes(procedure_dict_ctakes[filename], search_p))
# with open('./Data/procedure_files/procedure_vocab.pkl', 'wb') as handle:
#     pickle.dump(procedure_vocab, handle)

procedure_label_dict = dict()
procedure_mask_dict = dict()
for filename in tqdm(procedure_dict_ctakes.keys()):
    procedure_list = procedure_dict_ctakes[filename]
    procedure_list_filtered = filter_icd_codes(procedure_list, search_p)
    procedure_mask = get_mask(procedure_list_filtered, procedure_ctakes_out, search_p)
    procedure_mask_dict[filename] = procedure_mask
    procedure_final_list = add_parents(procedure_list_filtered, search_p)
    procedure_label_dict[filename] = procedure_final_list
#one hot encoding
#labels
filenames = list(procedure_label_dict.keys())
labels = list(procedure_label_dict.values())
p_counter = Counter()
for label_list in labels:
    p_counter.update(label_list)
keep = [x for x in list(set(p_counter)) if p_counter[x]>=50]
mlb = MultiLabelBinarizer()
mlb.fit([keep,])
one_hot_encoded = mlb.transform(labels)
print(len(mlb.classes_))
procedure_label_dict = {filename:label for filename,label in zip(filenames, one_hot_encoded)}

#mask
filenames = list(procedure_mask_dict.keys())
mask = list(procedure_mask_dict.values())
one_hot_encoded = mlb.transform(mask)
one_hot_encoded = 1 - one_hot_encoded
procedure_mask_dict = {filename:m for filename,m in zip(filenames, one_hot_encoded)}

with open('./Data/procedure_files/procedure_label_dict_ext.pkl', 'wb') as handle:
    pickle.dump(procedure_label_dict, handle)
with open('./Data/procedure_files/procedure_mask_dict_ext.pkl', 'wb') as handle:
    pickle.dump(procedure_mask_dict, handle)
with open('./Data/procedure_files/procedure_label_encoder_ext.pkl', 'wb') as handle:
    pickle.dump(mlb, handle)

#medicine label
filenames = list(medicine_dict_ctakes.keys())
labels = list(medicine_dict_ctakes.values())
m_counter = Counter()
for label_list in labels:
    m_counter.update(label_list)
keep = [x for x in list(set(m_counter)) if m_counter[x]>=5]
mlb = MultiLabelBinarizer()
mlb.fit([keep,])

#find similar medicines from drugs.com
medicine_labels = [preprocess_medicine_name(medicine_name) for medicine_name in mlb.classes_]
medicine_files = glob.glob('./Data/JSON/*')
similar_medicine_dict = defaultdict(list)
for filepath in tqdm(medicine_files[100:]):
#     print(filepath)
    medicine_name = filepath.split('/')[-1].split('.')[0]
    medicine_name = preprocess_medicine_name(medicine_name)
    if(medicine_name not in medicine_labels):
        continue
    with open(filepath, 'r') as file:
        datastore = json.load(file)
        if(datastore['Related drugs'] != {}):
            for key in datastore['Related drugs'].keys():
                similar_medicine_dict[medicine_name].extend(datastore['Related drugs'][key])
            similar_medicine_dict[medicine_name] = list(set(similar_medicine_dict[medicine_name]).intersection(set(medicine_labels)))
        # if(len(similar_medicine_dict[medicine_name]) == 0):
        #     print(medicine_name, filepath)
with open('./Data/medicine_files/similar_medicine_dict_ext.pkl', 'wb') as handle:
    pickle.dump(similar_medicine_dict, handle)

medicines_filtered = list(similar_medicine_dict.keys())
mlb.fit([medicines_filtered,])

labels = [[preprocess_medicine_name(medicine_name) for medicine_name in label_list] for label_list in labels]
one_hot_encoded = mlb.transform(labels)
print(len(mlb.classes_))
medicine_label_dict = {filename:label for filename,label in zip(filenames, one_hot_encoded)}
with open('./Data/medicine_files/medicine_label_dict_ext.pkl', 'wb') as handle:
    pickle.dump(medicine_label_dict, handle)
with open('./Data/medicine_files/medicine_label_encoder_ext.pkl', 'wb') as handle:
    pickle.dump(mlb, handle)
