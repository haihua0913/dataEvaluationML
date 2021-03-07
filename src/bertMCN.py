#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/7/21 11:19 AM
# @Author  : Haihua
# @Email   : haihua.chen@unt.edu
# @Site    : 
# @File    : bertMCN.py
# @Software: PyCharm


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tensorflow as tf
from transformers import AutoModel
import time
from sklearn.metrics import classification_report

# test whether the code is running on GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# test the GPU version
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0)
print(gpu_name)

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
CUDA_LAUNCH_BLOCKING=1

# number of batch_size and training epochs
batch_size, epochs = 32, 16

def readolder(path):
    files = os.listdir(path)
    files.sort()
    return files


'''
Input the csv file, return the bert input
'''
def data_transform(filename):
    column_name = ['label', 'concept']
    df = pd.read_csv(filename,header=None,names=column_name)
    # Create sentence and label lists
    concepts = df.concept.values
    # We need to add special tokens at the beginning and end of each sentence(here concept) for BERT to work properly
    concepts = ["[CLS] " + str(concept) + " [SEP]" for concept in concepts]
    labels = df.label.values
    unique_labels = list(set(labels))
    print(unique_labels)
    print(len(unique_labels))
    return concepts,labels,unique_labels


'''
Tokenization
'''
def bert_representation(filename):
    concepts, labels, unique_labels = data_transform(filename)
    # import the BERT tokenizer, used to convert our concepts into tokens that correspond to BERT's vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_concepts_full = [tokenizer.tokenize(concept) for concept in concepts]
    print(tokenized_concepts_full)
    # Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
    # In the original paper, the authors used a length of 512.
    MAX_LEN = 512
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_concepts_full]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return input_ids,labels,attention_masks


'''
Read data from local
'''
# Read training data, validation data, and test data
# for i in range(0,10) indicate 10 folders
train_file_folder = '/home/isiia/PycharmProjects/concept_normalization/mcn-tweet-experiment/Twadr_L/train'
valid_file_folder = '/home/isiia/PycharmProjects/concept_normalization/mcn-tweet-experiment/Twadr_L/valid'
test_file_folder = '/home/isiia/PycharmProjects/concept_normalization/mcn-tweet-experiment/Twadr_L/test'
train_files = readolder(train_file_folder)
valid_files = readolder(valid_file_folder)
test_files = readolder(test_file_folder)
train_filename = train_file_folder+'/'+'TwADR-L.fold-0.train.csv'
valid_filename = valid_file_folder+'/'+'TwADR-L.fold-0.validation.csv'
test_filename = valid_file_folder+'/'+'TwADR-L.fold-0.test.csv'
train_inputs, train_labels, train_masks=bert_representation(train_filename)
validation_inputs, validation_labels, validation_masks = bert_representation(valid_filename)
test_inputs, test_labels, test_masks = bert_representation(test_filename)


'''
Convert Integer Sequences to Tensors
'''
# for train set
train_seq = torch.tensor(train_inputs)
train_mask = torch.tensor(train_masks)
train_y = torch.tensor(train_labels)

# for validation set
val_seq = torch.tensor(validation_inputs)
val_mask = torch.tensor(validation_masks)
val_y = torch.tensor(validation_labels)

# for test set
test_seq = torch.tensor(test_inputs)
test_mask = torch.tensor(test_masks)
test_y = torch.tensor(test_labels)


'''
Create DataLoaders
'''
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)


'''
Import BERT Model
Freeze BERT Parameters
# https://huggingface.co/gsarti/scibert-nli
'''
bert = AutoModel.from_pretrained('bert-base-uncased')
# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False


'''
Define Model Architecture
'''
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768, 512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
# push the model to GPU
model = model.to(device)

# optimizer from hugging face transformers
from transformers import AdamW
# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)


'''
Find Class Weights
'''
from sklearn.utils.class_weight import compute_class_weight
#compute the class weights
class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)
print(class_wts)
# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
# loss function
cross_entropy  = nn.NLLLoss(weight=weights)


'''
Fine-Tune BERT
'''
# function to train the model
def train():
    model.train()
    total_loss, total_accuracy = 0, 0
    # empty list to save model predictions
    total_preds = []
    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        # clear previously calculated gradients
        model.zero_grad()
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds)
    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    # returns the loss and predictions
    return avg_loss, total_preds

# function for evaluating the model
def evaluate():
    print("\nEvaluating...")
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    # empty list to save the model predictions
    total_preds = []
    # iterate over batches
    # t0 = time.time()
    for step, batch in enumerate(val_dataloader):
        t0 = time.time()
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = time.time()
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)
            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


'''
Start Model Training
'''
# set initial loss to infinite
best_valid_loss = float('inf')
# empty lists to store training and validation loss of each epoch
train_losses = []
valid_losses = []
# for each epoch
for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    # train model
    train_loss, _ = train()
    # evaluate model
    valid_loss, _ = evaluate()
    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


'''
Load Saved Model
'''
#load weights of best model
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))


'''
Get Predictions for Test Data
'''
# get predictions for test data
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()

# model's performance
preds = np.argmax(preds, axis = 1)
predict_list = preds.tolist()
print(predict_list)
print(classification_report(test_y, preds))

# confusion matrix
pd.crosstab(test_y, preds)