import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from Networks.han.model import HierarchialAttentionNetwork as HAN_net
from icd9cms import search as search_d
from icd9pcs import search as search_p
import pickle

alt_data_dir = 'Data'

def get_adj_matrices(search, mlb):
    '''
    Using ICD hierarchy create adjacency matrix disease and procedure codes
    '''
    classes = mlb.classes_
    A_in = np.zeros((len(classes), len(classes)))
    A_out = np.zeros((len(classes), len(classes)))
    A_self = np.identity(len(classes))
    for i in range(len(classes)):
        children = search(classes[i]).children
        children = [child.alt_code for child in children]
        children = list(set(children).intersection(set(classes)))
        if(len(children)>0):
            A_out[i,:] = mlb.transform([children])/len(children)
        parent = search(classes[i]).parent
        if((parent is not None) and (parent.alt_code in classes)):
            A_in[i,:] = mlb.transform([[parent.alt_code,]])
    return torch.from_numpy(A_in).float().cuda(), torch.from_numpy(A_out).float().cuda(), torch.from_numpy(A_self).float().cuda()

def get_adj_matrix_medicine(mlb):
    '''
    Return adjacency matrix corresponding to the drug graph created by connecting medicines used to treat same disease
    '''
    classes = mlb.classes_
    similar_medicine_dict = pickle.load(open(f'./Data/medicine_files/similar_medicine_dict_ext.pkl', 'rb'))
    A = np.zeros((len(classes), len(classes)))
    A_self = np.identity(len(classes))
    for i in range(len(classes)):
        if(len(similar_medicine_dict[classes[i]]) > 0):
            A[i,:] = mlb.transform([similar_medicine_dict[classes[i]]])/len(similar_medicine_dict[classes[i]])
            A[i,i] = 0
    return torch.from_numpy(A).float().cuda(), torch.from_numpy(A_self).float().cuda()

class multiclassifier(nn.Module):
    '''
    Returns the procedure, disease and medicine embeddings given out by GCN and the similarity scores.
    '''
    def __init__(self, han_net_params, feature_dim, dropout=0.2):
        super().__init__()
        self.han_net = HAN_net(**han_net_params)
        self.linear = nn.Linear(feature_dim, 300)
        self.dropout = nn.Dropout(dropout)
        self.diagnoses_mlb = pickle.load(open(f'./{alt_data_dir}/diagnoses_files/diagnoses_label_encoder_ext.pkl', 'rb'))
        self.procedure_mlb = pickle.load(open(f'./{alt_data_dir}/procedure_files/procedure_label_encoder_ext.pkl', 'rb'))
        self.medicine_mlb = pickle.load(open(f'./Data/medicine_files/medicine_label_encoder_ext.pkl', 'rb'))

        # self.diagnoses_embeddings_init = torch.randn(len(self.diagnoses_mlb.classes_),512).float().cuda()
        temp = torch.arange(0, len(self.diagnoses_mlb.classes_)).cuda() # set size (2,10) for MHE
        self.diagnoses_embeddings_init = torch.zeros(len(self.diagnoses_mlb.classes_), len(self.diagnoses_mlb.classes_)).cuda()
        self.diagnoses_embeddings_init[temp, temp] = 1
        self.linear_diagnoses_in_1 = nn.Linear(len(self.diagnoses_mlb.classes_), 300)
        self.linear_diagnoses_out_1 = nn.Linear(len(self.diagnoses_mlb.classes_), 300)
        self.linear_diagnoses_self_1 = nn.Linear(len(self.diagnoses_mlb.classes_), 300)
        self.linear_diagnoses_in_2 = nn.Linear(300, 300)
        self.linear_diagnoses_out_2 = nn.Linear(300, 300)
        self.linear_diagnoses_self_2 = nn.Linear(300, 300)
        self.diagnoses_A_in, self.diagnoses_A_out, self.diagnoses_A_self = get_adj_matrices(search_d, self.diagnoses_mlb)

        # self.procedure_embeddings_init = torch.randn(len(self.procedure_mlb.classes_),512).cuda()
        temp = torch.arange(0, len(self.procedure_mlb.classes_)).cuda()
        self.procedure_embeddings_init = torch.zeros(len(self.procedure_mlb.classes_), len(self.procedure_mlb.classes_)).cuda()
        self.procedure_embeddings_init[temp, temp] = 1
        self.linear_procedure_in_1 = nn.Linear(len(self.procedure_mlb.classes_), 300)
        self.linear_procedure_out_1 = nn.Linear(len(self.procedure_mlb.classes_), 300)
        self.linear_procedure_self_1 = nn.Linear(len(self.procedure_mlb.classes_), 300)
        self.linear_procedure_in_2 = nn.Linear(300, 300)
        self.linear_procedure_out_2 = nn.Linear(300, 300)
        self.linear_procedure_self_2 = nn.Linear(300, 300)
        self.procedure_A_in, self.procedure_A_out, self.procedure_A_self = get_adj_matrices(search_p, self.procedure_mlb)

        # self.medicine_embeddings_init = torch.randn(len(self.medicine_mlb.classes_),512).cuda()
        temp = torch.arange(0, len(self.medicine_mlb.classes_)).cuda()
        self.medicine_embeddings_init = torch.zeros(len(self.medicine_mlb.classes_), len(self.medicine_mlb.classes_)).cuda()
        self.medicine_embeddings_init[temp, temp] = 1
        self.linear_medicine_1 = nn.Linear(len(self.medicine_mlb.classes_), 300)
        self.linear_medicine_self_1 = nn.Linear(len(self.medicine_mlb.classes_), 300)
        self.linear_medicine_2 = nn.Linear(300, 300)
        self.linear_medicine_self_2 = nn.Linear(300, 300)
        self.medicine_A, self.medicine_A_self = get_adj_matrix_medicine(self.medicine_mlb)

    def forward(self, han_net_inputs):
        ds_embedding_1, _, _, _ = self.han_net(**han_net_inputs)
        ds_embedding = self.linear(ds_embedding_1)
        outs = []

    
        diagnoses_embeddings = torch.mm(self.diagnoses_A_in, self.linear_diagnoses_in_1(self.diagnoses_embeddings_init))+\
                                torch.mm(self.diagnoses_A_self, self.linear_diagnoses_self_1(self.diagnoses_embeddings_init))+\
                                torch.mm(self.diagnoses_A_out, self.linear_diagnoses_out_1(self.diagnoses_embeddings_init))
        diagnoses_out = ds_embedding.unsqueeze(1)*diagnoses_embeddings.unsqueeze(0)
        outs.append(torch.sum(diagnoses_out, dim=2))

        procedure_embeddings = torch.mm(self.procedure_A_in, self.linear_procedure_in_1(self.procedure_embeddings_init))+\
                                torch.mm(self.procedure_A_self, self.linear_procedure_self_1(self.procedure_embeddings_init))+\
                                torch.mm(self.procedure_A_out, self.linear_procedure_out_1(self.procedure_embeddings_init))
        procedure_out = ds_embedding.unsqueeze(1)*procedure_embeddings.unsqueeze(0)
        outs.append(torch.sum(procedure_out, dim=2))

        medicine_embeddings = torch.mm(self.medicine_A, self.linear_medicine_1(self.medicine_embeddings_init))+\
                                torch.mm(self.medicine_A_self, self.linear_medicine_self_1(self.medicine_embeddings_init))
        medicine_out = ds_embedding.unsqueeze(1)*medicine_embeddings.unsqueeze(0)
        outs.append(torch.sum(medicine_out, dim=2))

        return torch.cat((ds_embedding, ds_embedding_1),1), diagnoses_embeddings, procedure_embeddings, medicine_embeddings, outs
