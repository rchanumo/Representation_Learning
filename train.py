import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from model import DSN
from generateDataIters import dataiter
from Networks.han.utils import *
from progiter import ProgIter as tqdm
import itertools
import heapq
import argparse
import sys
import pickle
import warnings
from icd9cms.icd9 import search
from multiclassifier import multiclassifier
import sklearn
from sklearn.metrics import hamming_loss, average_precision_score, f1_score
from class_balanced_loss import CB_loss

parser = argparse.ArgumentParser()
parser.add_argument("--expname", type=str, help='name of experiment', default='1')
args = parser.parse_args()

mainiter = dataiter()
train_iter, valid_iter, singular_iter, TEXT, FILENAME = mainiter.get_iters()
print('Loaded data**********\n\n')

han_net_params = {
'n_classes':1,
'vocab_size':len(TEXT.vocab),
'emb_size':200,
'word_rnn_size':512,
'sentence_rnn_size':1024,
'word_rnn_layers':1,
'sentence_rnn_layers':1,
'word_att_size':512,
'sentence_att_size':1024,
'dropout':0.3,
'attn':1
}
han_net_extras = {
'embeddings':TEXT.vocab.vectors,
'fine_tune_word_embeddings':True
}

# Training parameters
start_epoch = 0 
lr = 1e-3
momentum = 0.9
epochs = 20
grad_clip = None
print_freq = 200
model_name = args.expname
checkpoint = f'./Models/han/checkpoint_{model_name}.pth.tar'
best_acc = 0.
epochs_since_improvement=0

use_cuda = torch.cuda.is_available()

alt_data_dir = 'Data'

diagnoses_label_dict = pickle.load(open(f'./{alt_data_dir}/diagnoses_files/diagnoses_label_dict_ext.pkl', 'rb'))
diagnoses_mask_dict = pickle.load(open(f'./{alt_data_dir}/diagnoses_files/diagnoses_mask_dict_ext.pkl', 'rb'))

procedure_label_dict = pickle.load(open(f'./{alt_data_dir}/procedure_files/procedure_label_dict_ext.pkl', 'rb'))
procedure_mask_dict = pickle.load(open(f'./{alt_data_dir}/procedure_files/procedure_mask_dict_ext.pkl', 'rb'))

medicine_label_dict = pickle.load(open(f'./Data/medicine_files/medicine_label_dict_ext.pkl', 'rb'))

def main():
    """
    Training and validation.
    """
    global best_acc, epochs_since_improvement, checkpoint, start_epoch

    # Initialize model or load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        # word_map = checkpoint['word_map']
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        print(
            '\nLoaded checkpoint from epoch %d, with a previous best accuracy of %.3f.\n' % (start_epoch - 1, best_acc))
    else:
        diagnoses_mlb = pickle.load(open(f'./{alt_data_dir}/diagnoses_files/diagnoses_label_encoder_ext.pkl', 'rb'))
        procedure_mlb = pickle.load(open(f'./{alt_data_dir}/procedure_files/procedure_label_encoder_ext.pkl', 'rb'))
        medicine_mlb = pickle.load(open(f'./Data/medicine_files/medicine_label_encoder_ext.pkl', 'rb'))
        model = multiclassifier(han_net_params, han_net_params['sentence_rnn_size']*2)#, [len(diagnoses_mlb.classes_), len(procedure_mlb.classes_), len(medicine_mlb.classes_)])
        model.han_net.sentence_attention.word_attention.init_embeddings(
            han_net_extras['embeddings'])  # initialize embedding layer with pre-trained embeddings
        model.han_net.sentence_attention.word_attention.fine_tune_embeddings(han_net_extras['fine_tune_word_embeddings'])  # fine-tune
        optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Loss functions
    # diagnoses_label_values = np.vstack(list(diagnoses_label_dict.values()))
    # diagnoses_weights = len(diagnoses_label_values) - np.sum(diagnoses_label_values, axis=0)
    # diagnoses_weights = diagnoses_weights/np.sum(diagnoses_label_values, axis=0)
    # print(diagnoses_weights)
    criterion_d = nn.BCEWithLogitsLoss(reduction='none')#, pos_weight=torch.from_numpy(diagnoses_weights).float())

    # procedure_label_values = np.vstack(list(procedure_label_dict.values()))
    # procedure_weights = len(procedure_label_values) -  np.sum(procedure_label_values, axis=0)
    # procedure_weights = procedure_weights/np.sum(procedure_label_values, axis=0)
    # print(procedure_weights)
    criterion_p = nn.BCEWithLogitsLoss(reduction='none')#, pos_weight=torch.from_numpy(procedure_weights).float())

    # medicine_label_values = np.vstack(list(medicine_label_dict.values()))
    # medicine_weights = len(medicine_label_values) - np.sum(medicine_label_values, axis=0)
    # medicine_weights = medicine_weights/np.sum(medicine_label_values, axis=0)
    # print(medicine_weights)
    criterion_m = nn.BCEWithLogitsLoss(reduction='none')#, pos_weight=torch.from_numpy(medicine_weights).float())

    #get label freqs
    # label_size_section_wise = {'d_label_size': np.sum(diagnoses_label_values, axis=0).tolist(),
    #                             'p_label_size': np.sum(procedure_label_values, axis=0).tolist(),
    #                             'm_label_size': np.sum(medicine_label_values, axis=0).tolist()}
    label_size_section_wise = None
   
#     # Move to device
#     if torch.cuda.device_count() > 1:
#           print("Let's use", torch.cuda.device_count(), "GPUs!")
#           # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#           model = torch.nn.DataParallel(model)
    if(use_cuda):
        model = model.cuda()#to(device)
        criterion_d = criterion_d.cuda()#to(device)
        criterion_p = criterion_p.cuda()#to(device)
        criterion_m = criterion_m.cuda()#to(device)

    #use multiple gpus
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = torch.nn.DataParallel(model)

    get_document_embeddings(data_loader=singular_iter, model=model)
    # get_label_embeddings(data_loader=singular_iter, model=model)
    assert 1==2
    # Epochs
    best_acc = 0
    for epoch in range(start_epoch, start_epoch+epochs):
        # #One epoch's training
        # print('*************************Training**********************************')
        start_time = time.time()
        acc = train_model(train_loader=train_iter,
                        model=model,
                        criterion_d=criterion_d,
                        criterion_p=criterion_p,
                        criterion_m=criterion_m,
                        optimizer=optimizer,
                        epoch=epoch,
                        label_size_section_wise=label_size_section_wise)
        print('Train Time: ', time.time()-start_time, '\n')
        # #One epoch's validation
        # print('*************************Testing************************************')
        # start_time = time.time()
        # acc = validate_model(val_loader=valid_iter,
                        # model=model,
                        # criterion_d=criterion_d,
                        # criterion_p=criterion_p,
                        # criterion_m=criterion_m,
                        # optimizer=optimizer,
                        # epoch=epoch,
                        # label_size_section_wise=label_size_section_wise)
        # print('Validation Time: ', time.time()-start_time, '\n')
        # Did validation accuracy improve?
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # epochs_since_improvement = 0
        # best_acc = 1.0
        # is_best = False
        # #Decay learning rate every epoch
        # # adjust_learning_rate(optimizer, 0.5)

        #Save checkpoint
        save_checkpoint(epoch, model, optimizer, best_acc, epochs_since_improvement, is_best, 'han', model_name)

        sys.stdout.flush()


def precision_at_n(Y_true, Y_scores, n=8):
    precision = 0
    for i in range(len(Y_scores)):
        y_scores = Y_scores[i,:]
        y_true = Y_true[i,:]
        y_true = np.argwhere(y_true==1)
        y_predicted = np.argsort(y_scores)
        y_predicted = y_predicted[-n:]
        precision += len(np.intersect1d(y_true, y_predicted))/n
    return precision/len(Y_scores)

def train_model(train_loader, model, criterion_d, criterion_p, criterion_m, optimizer, epoch, label_size_section_wise, beta = 0.9999, gamma = 2.0):
    """
    Performs one epoch's training.
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    # diagnoses_predictions_epoch = []
    # diagnoses_labels_epoch = []
    # procedure_predictions_epoch = []
    # procedure_labels_epoch = []
    # medicine_predictions_epoch = []
    # medicine_labels_epoch = []
    diagnoses_AP = []
    diagnoses_f1 = []
    diagnoses_pn = []
    medicine_AP = []
    medicine_f1 = []
    medicine_pn = []
    procedure_AP = []
    procedure_f1 = []
    procedure_pn = []

    start = time.time()

    sigmoid = torch.nn.Sigmoid()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=sklearn.exceptions.UndefinedMetricWarning)
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        warnings.filterwarnings("ignore",category=UserWarning)
        # Batches
        start_time = time.time()
        for i, batch in enumerate(train_loader):

            ds, sentences_per_document_ds, words_per_sentence_ds = batch.ds
            # print('size of batch: ', ds.shape)
            # continue
            if(use_cuda):
                ds, sentences_per_document_ds, words_per_sentence_ds = ds.cuda(), sentences_per_document_ds.cuda(), words_per_sentence_ds.cuda()
            han_net_ds_inputs = {
            'documents':ds,
            'sentences_per_document':sentences_per_document_ds,
            'words_per_sentence':words_per_sentence_ds
            }
            filenames = FILENAME.reverse(batch.filename.unsqueeze(0))

            diagnoses_labels = [diagnoses_label_dict[filename] for filename in filenames]
            diagnoses_labels = np.vstack(diagnoses_labels)
            diagnoses_labels = torch.from_numpy(diagnoses_labels).float()

            diagnoses_mask = [diagnoses_mask_dict[filename] for filename in filenames]
            diagnoses_mask = np.vstack(diagnoses_mask)
            diagnoses_mask = torch.from_numpy(diagnoses_mask).float()

            procedure_labels = [procedure_label_dict[filename] for filename in filenames]
            procedure_labels = np.vstack(procedure_labels)
            procedure_labels = torch.from_numpy(procedure_labels).float()

            procedure_mask = [procedure_mask_dict[filename] for filename in filenames]
            procedure_mask = np.vstack(procedure_mask)
            procedure_mask = torch.from_numpy(procedure_mask).float()


            medicine_labels = [medicine_label_dict[filename] for filename in filenames]
            medicine_labels = np.vstack(medicine_labels)
            medicine_labels = torch.from_numpy(medicine_labels).float()

            if(use_cuda):
                diagnoses_labels = diagnoses_labels.cuda()
                diagnoses_mask = diagnoses_mask.cuda()
                procedure_labels = procedure_labels.cuda()
                procedure_mask = procedure_mask.cuda()
                medicine_labels = medicine_labels.cuda()

            data_time.update(time.time() - start)

            # Forward prop.
    #         print(documents.size(), onet_desc.size())
            embeddings, diagnoses_embeddings, procedure_embeddings, medicine_embeddings, out = model(han_net_ds_inputs)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)
            # Loss
    #         print(similarities.size(), similarities.type(), labels.size(), labels.type())

            diagnoses_loss = criterion_d(out[0], diagnoses_labels)  # scalar
            # diagnoses_loss = CB_loss(diagnoses_labels, out[0], label_size_section_wise['d_label_size'], len(label_size_section_wise['d_label_size']), 'focal', beta, gamma)
            diagnoses_loss = diagnoses_loss*diagnoses_mask
            diagnoses_loss =  torch.sum(diagnoses_loss)/diagnoses_loss.size(1)

            procedure_loss = criterion_p(out[1], procedure_labels)
            # procedure_loss = CB_loss(procedure_labels, out[1], label_size_section_wise['p_label_size'], len(label_size_section_wise['p_label_size']), 'focal', beta, gamma)
            procedure_loss = procedure_loss*procedure_mask
            procedure_loss = torch.sum(procedure_loss)/procedure_loss.size(1)

            medicine_loss = criterion_m(out[2] , medicine_labels)
            # medicine_loss = CB_loss(medicine_labels, out[2], label_size_section_wise['m_label_size'], len(label_size_section_wise['m_label_size']), 'focal', beta, gamma)
            medicine_loss = torch.sum(medicine_loss)/medicine_loss.size(1)

            loss = diagnoses_loss + procedure_loss + medicine_loss
            # loss = medicine_loss
            # Back prop.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            # Update
            optimizer.step()

            # Find accuracy
            
            diagnoses_predictions_epoch = sigmoid(out[0]).data.cpu().numpy()
            diagnoses_labels_epoch = diagnoses_labels.type(torch.LongTensor).data.cpu().numpy()
            diagnoses_AP.append(average_precision_score(diagnoses_labels_epoch, diagnoses_predictions_epoch))
            diagnoses_f1.append(f1_score(diagnoses_labels_epoch, np.array(diagnoses_predictions_epoch>0.5, dtype=int), average='macro'))
            diagnoses_pn.append(precision_at_n(diagnoses_labels_epoch, diagnoses_predictions_epoch))

            procedure_predictions_epoch = sigmoid(out[1]).data.cpu().numpy()
            procedure_labels_epoch = procedure_labels.type(torch.LongTensor).data.cpu().numpy()
            procedure_AP.append(average_precision_score(procedure_labels_epoch, procedure_predictions_epoch))
            procedure_f1.append(f1_score(procedure_labels_epoch, np.array(procedure_predictions_epoch>0.5, dtype=int), average='macro'))
            procedure_pn.append(precision_at_n(procedure_labels_epoch, procedure_predictions_epoch))

            medicine_predictions_epoch = sigmoid(out[2]).data.cpu().numpy()
            medicine_labels_epoch = medicine_labels.type(torch.LongTensor).data.cpu().numpy()
            medicine_AP.append(average_precision_score(medicine_labels_epoch, medicine_predictions_epoch))
            medicine_f1.append(f1_score(medicine_labels_epoch, np.array(medicine_predictions_epoch>0.5, dtype=int), average='macro'))
            medicine_pn.append(precision_at_n(medicine_labels_epoch, medicine_predictions_epoch))

            # Keep track of metrics
            losses.update(loss.item(), len(ds))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print training status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, 0,
                                                                    batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
                print('Time passed so far: ', time.time()-start_time)
    print('\n *  Train LOSS - {loss.avg:.3f}\n'.format(loss=losses))
    # diagnoses_predictions_epoch = np.concatenate(diagnoses_predictions_epoch)
    # diagnoses_labels_epoch = np.concatenate(diagnoses_labels_epoch)
    # procedure_predictions_epoch = np.concatenate(procedure_predictions_epoch)
    # procedure_labels_epoch = np.concatenate(procedure_labels_epoch)
    # medicine_predictions_epoch = np.concatenate(medicine_predictions_epoch)
    # medicine_labels_epoch = np.concatenate(medicine_labels_epoch)
    # print('Diagnoses AP: ', average_precision_score(diagnoses_labels_epoch, diagnoses_predictions_epoch))
    # print('Procedure AP: ', average_precision_score(procedure_labels_epoch, procedure_predictions_epoch))
    # print('medicine AP: ', average_precision_score(medicine_labels_epoch, medicine_predictions_epoch))
    # print('Diagnoses f1: ', f1_score(diagnoses_labels_epoch, np.array(diagnoses_predictions_epoch>0.5, dtype=int), average='macro'))
    # print('Procedure f1: ', f1_score(procedure_labels_epoch, np.array(procedure_predictions_epoch>0.5, dtype=int), average='macro'))
    # print('medicine f1: ', f1_score(medicine_labels_epoch, np.array(medicine_predictions_epoch>0.5, dtype=int), average='macro'))
    # print('Diagnoses p@n: ', precision_at_n(diagnoses_labels_epoch, diagnoses_predictions_epoch))
    # print('Procedure p@n: ', precision_at_n(procedure_labels_epoch, procedure_predictions_epoch))
    # print('medicine p@n: ', precision_at_n(medicine_labels_epoch, medicine_predictions_epoch))
    print('Diagnoses AP: ', sum(diagnoses_AP)/len(diagnoses_AP))
    print('Procedure AP: ', sum(procedure_AP)/len(procedure_AP))
    print('medicine AP: ', sum(medicine_AP)/len(medicine_AP))
    print('Diagnoses f1: ', sum(diagnoses_f1)/len(diagnoses_f1))
    print('Procedure f1: ', sum(procedure_f1)/len(procedure_f1))
    print('medicine f1: ', sum(medicine_f1)/len(medicine_f1))
    print('Diagnoses p@n: ', sum(diagnoses_pn)/len(diagnoses_pn))
    print('Procedure p@n: ', sum(procedure_pn)/len(procedure_pn))
    print('medicine p@n: ', sum(medicine_pn)/len(medicine_pn))
    return sum(diagnoses_pn)/len(diagnoses_pn)


def validate_model(val_loader, model, criterion_d, criterion_p, criterion_m, optimizer, epoch, label_size_section_wise):
    """
    Performs one epoch's validation.
    """

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter() 
    losses = AverageMeter()
    diagnoses_AP = []
    diagnoses_f1 = []
    diagnoses_pn = []
    medicine_AP = []
    medicine_f1 = []
    medicine_pn = []
    procedure_AP = []
    procedure_f1 = []
    procedure_pn = []

    start = time.time()

    sigmoid = torch.nn.Sigmoid()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=sklearn.exceptions.UndefinedMetricWarning)
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        warnings.filterwarnings("ignore",category=UserWarning)
        # Batches
        start_time = time.time()
        for i, batch in enumerate(val_loader):

            ds, sentences_per_document_ds, words_per_sentence_ds = batch.ds
            # print('size of batch: ', ds.shape)
            # continue
            if(use_cuda):
                ds, sentences_per_document_ds, words_per_sentence_ds = ds.cuda(), sentences_per_document_ds.cuda(), words_per_sentence_ds.cuda()
            han_net_ds_inputs = {
            'documents':ds,
            'sentences_per_document':sentences_per_document_ds,
            'words_per_sentence':words_per_sentence_ds
            }
            filenames = FILENAME.reverse(batch.filename.unsqueeze(0))

            diagnoses_labels = [diagnoses_label_dict[filename] for filename in filenames]
            diagnoses_labels = np.vstack(diagnoses_labels)
            diagnoses_labels = torch.from_numpy(diagnoses_labels).float()

            diagnoses_mask = [diagnoses_mask_dict[filename] for filename in filenames]
            diagnoses_mask = np.vstack(diagnoses_mask)
            diagnoses_mask = torch.from_numpy(diagnoses_mask).float()

            procedure_labels = [procedure_label_dict[filename] for filename in filenames]
            procedure_labels = np.vstack(procedure_labels)
            procedure_labels = torch.from_numpy(procedure_labels).float()

            procedure_mask = [procedure_mask_dict[filename] for filename in filenames]
            procedure_mask = np.vstack(procedure_mask)
            procedure_mask = torch.from_numpy(procedure_mask).float()


            medicine_labels = [medicine_label_dict[filename] for filename in filenames]
            medicine_labels = np.vstack(medicine_labels)
            medicine_labels = torch.from_numpy(medicine_labels).float()

            if(use_cuda):
                diagnoses_labels = diagnoses_labels.cuda()
                diagnoses_mask = diagnoses_mask.cuda()
                procedure_labels = procedure_labels.cuda()
                procedure_mask = procedure_mask.cuda()
                medicine_labels = medicine_labels.cuda()

            data_time.update(time.time() - start)

            # Forward prop.
    #         print(documents.size(), onet_desc.size())
            embeddings, diagnoses_embeddings, procedure_embeddings, medicine_embeddings, out = model(han_net_ds_inputs)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)
            # Loss
    #         print(similarities.size(), similarities.type(), labels.size(), labels.type())

            diagnoses_loss = criterion_d(out[0], diagnoses_labels)  # scalar
            # diagnoses_loss = CB_loss(diagnoses_labels, out[0], label_size_section_wise['d_label_size'], len(label_size_section_wise['d_label_size']), 'focal', beta, gamma)
            diagnoses_loss = diagnoses_loss*diagnoses_mask
            diagnoses_loss =  torch.sum(diagnoses_loss)/diagnoses_loss.size(1)

            procedure_loss = criterion_p(out[1], procedure_labels)
            # procedure_loss = CB_loss(procedure_labels, out[1], label_size_section_wise['p_label_size'], len(label_size_section_wise['p_label_size']), 'focal', beta, gamma)
            procedure_loss = procedure_loss*procedure_mask
            procedure_loss = torch.sum(procedure_loss)/procedure_loss.size(1)

            medicine_loss = criterion_m(out[2] , medicine_labels)
            # medicine_loss = CB_loss(medicine_labels, out[2], label_size_section_wise['m_label_size'], len(label_size_section_wise['m_label_size']), 'focal', beta, gamma)
            medicine_loss = torch.sum(medicine_loss)/medicine_loss.size(1)

            loss = diagnoses_loss + procedure_loss + medicine_loss
            # loss = medicine_loss

            # Find accuracy
            diagnoses_predictions_epoch = sigmoid(out[0]).data.cpu().numpy()
            diagnoses_labels_epoch = diagnoses_labels.type(torch.LongTensor).data.cpu().numpy()
            diagnoses_AP.append(average_precision_score(diagnoses_labels_epoch, diagnoses_predictions_epoch))
            diagnoses_f1.append(f1_score(diagnoses_labels_epoch, np.array(diagnoses_predictions_epoch>0.5, dtype=int), average='macro'))
            diagnoses_pn.append(precision_at_n(diagnoses_labels_epoch, diagnoses_predictions_epoch))

            procedure_predictions_epoch = sigmoid(out[1]).data.cpu().numpy()
            procedure_labels_epoch = procedure_labels.type(torch.LongTensor).data.cpu().numpy()
            procedure_AP.append(average_precision_score(procedure_labels_epoch, procedure_predictions_epoch))
            procedure_f1.append(f1_score(procedure_labels_epoch, np.array(procedure_predictions_epoch>0.5, dtype=int), average='macro'))
            procedure_pn.append(precision_at_n(procedure_labels_epoch, procedure_predictions_epoch))

            medicine_predictions_epoch = sigmoid(out[2]).data.cpu().numpy()
            medicine_labels_epoch = medicine_labels.type(torch.LongTensor).data.cpu().numpy()
            medicine_AP.append(average_precision_score(medicine_labels_epoch, medicine_predictions_epoch))
            medicine_f1.append(f1_score(medicine_labels_epoch, np.array(medicine_predictions_epoch>0.5, dtype=int), average='macro'))
            medicine_pn.append(precision_at_n(medicine_labels_epoch, medicine_predictions_epoch))

            # Keep track of metrics
            losses.update(loss.item(), len(ds))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print Validation status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, 0,
                                                                    batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
                print('Time passed so far: ', time.time()-start_time)
    print('\n *  Valid LOSS - {loss.avg:.3f}\n'.format(loss=losses))
    print('Diagnoses AP: ', sum(diagnoses_AP)/len(diagnoses_AP))
    print('Procedure AP: ', sum(procedure_AP)/len(procedure_AP))
    print('medicine AP: ', sum(medicine_AP)/len(medicine_AP))
    print('Diagnoses f1: ', sum(diagnoses_f1)/len(diagnoses_f1))
    print('Procedure f1: ', sum(procedure_f1)/len(procedure_f1))
    print('medicine f1: ', sum(medicine_f1)/len(medicine_f1))
    print('Diagnoses p@n: ', sum(diagnoses_pn)/len(diagnoses_pn))
    print('Procedure p@n: ', sum(procedure_pn)/len(procedure_pn))
    print('medicine p@n: ', sum(medicine_pn)/len(medicine_pn))
    return sum(diagnoses_pn)/len(diagnoses_pn)



def get_document_embeddings(data_loader, model):
    """
    Performs one epoch's validation.
    """
    model.eval()

    document_emebddings_flat = dict()
    for i, batch in tqdm(enumerate(data_loader)):

        ds, sentences_per_document_ds, words_per_sentence_ds = batch.ds
        filename = FILENAME.reverse(batch.filename.unsqueeze(0))

        if(use_cuda):
           ds, sentences_per_document_ds, words_per_sentence_ds = ds.cuda(), sentences_per_document_ds.cuda(), words_per_sentence_ds.cuda()
        han_net_ds_inputs = {
        'documents':ds,
        'sentences_per_document':sentences_per_document_ds,
       'words_per_sentence':words_per_sentence_ds
        }

        data_time.update(time.time() - start)
        embeddings, _, _, _, _= model(han_net_ds_inputs)
        embeddings = embeddings.data.cpu().numpy()
        document_emebddings_flat[filename[0]] = embeddings[0]

    with open(f'./Data/document_embeddings_trans_{args.expname}.pkl', 'wb') as handle:
        pickle.dump(document_emebddings_flat, handle)

def get_label_embeddings(data_loader, model):
    """
    Returns embeddings for ICD codes and medicines
    """
    model.eval()

    document_emebddings_flat = dict()
    for i, batch in tqdm(enumerate(data_loader)):


        ds, sentences_per_document_ds, words_per_sentence_ds = batch.ds
        filename = FILENAME.reverse(batch.filename.unsqueeze(0))

        if(use_cuda):
           ds, sentences_per_document_ds, words_per_sentence_ds = ds.cuda(), sentences_per_document_ds.cuda(), words_per_sentence_ds.cuda()
        han_net_ds_inputs = {
        'documents':ds,
        'sentences_per_document':sentences_per_document_ds,
        'words_per_sentence':words_per_sentence_ds
        }

        data_time.update(time.time() - start)
        _, diagnoses_embeddings, procedure_embeddings, medicine_embeddings, _ = model(han_net_ds_inputs)
        break

    with open(f'./Data/label_embeddings/diagnoses_embeddings_{args.expname}.pkl', 'wb') as handle:
        pickle.dump(diagnoses_embeddings.data.cpu().numpy(), handle)
    with open(f'./Data/label_embeddings/procedure_embeddings_{args.expname}.pkl', 'wb') as handle:
        pickle.dump(procedure_embeddings.data.cpu().numpy(), handle)
    with open(f'./Data/label_embeddings/medicine_embeddings_{args.expname}.pkl', 'wb') as handle:
        pickle.dump(medicine_embeddings.data.cpu().numpy(), handle)


if __name__ == '__main__':
    main()
