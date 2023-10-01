#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import pickle
import random
import string
import argparse
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score

# seed
SEED = 2022
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# embedding mapping (dict).
embed = {}
with open("glove.6B.100d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        x = line.split()
        y = x[1:]
        embed[x[0]] = y

# NLTK.
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# # download stopwords and wordnet.
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# preprocessing functions.
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# def remove_numbers(text):
#     """function to remove numbers"""
#     return 

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# filenames of data splits.
val_filename = "data/balanced_review_hijack_val.tsv"
test_filename = "data/balanced_review_hijack_test.tsv"
train_filename = "data/balanced_review_hijack_train.tsv"
# val_filename = "data/emanual_balanced_review_hijack_val.tsv"
# test_filename = "data/emanual_balanced_review_hijack_test.tsv"
# train_filename = "data/emanual_balanced_review_hijack_train.tsv"
# load tsv data.
val_df = pd.read_csv(val_filename, sep = '\t')
test_df = pd.read_csv(test_filename, sep = '\t')
train_df = pd.read_csv(train_filename, sep = '\t')
dfs = [train_df, test_df, val_df]
# mapping of classes.
class_mapping = {'-': 0, "sbsc": 1, "sbdc": 2, "dbsc": 3, "dbdc": 4}
print(train_df.head())
# print(train_df["class"][:5])
def preproc(s: str) -> str:
    s = str(s).lower()
    s = remove_stopwords(s)
    s = remove_punctuation(s)
    # s = remove_numbers(s)
    s = lemmatize_words(s)
    
    return s
    
for i in range(len(dfs)):
    dfs[i]["product"] = dfs[i]["product"].apply(lambda s: preproc(s))
    dfs[i]["review"] = dfs[i]["review"].apply(lambda s: preproc(s))
    # dfs[i]["class"] = dfs[i]["class"].apply(lambda c: class_mapping[c])
print(train_df.head())
# create the set of unique words.
word_set = set()
for i in range(len(dfs)):
    for prod, review in zip(dfs[i]["product"], dfs[i]["review"]):
        for word in prod.split():
            word_set.add(word)
        for word in review.split():
            word_set.add(word)
# create word to index mapping.
vocab2index = {"": 0, "UNK": 1}
words = ["", "UNK"]
for word in word_set:
    vocab2index[word] = len(words)
    words.append(word)
# create index to word mapping.
index2vocab = {v: k for k,v in vocab2index.items()}
# function to encode sentence.
def encode_sentence(text: str, vocab2index: dict, N: int=100):
    tokenized = text.split()
    # start with zeros (this makes sure input is padded to N).
    encoded = np.zeros(N, dtype=int)
    # create numpy array of input ids.
    input_ids = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    # length of sequences.
    length = min(N, len(input_ids))
    # set input ids for the np.zeros array.
    encoded[:length] = input_ids[:length]
    # create mask.
    mask = (encoded > 0).astype(int)
    
    return encoded, length, mask

for i in range(len(dfs)):
    dfs[i]["encoded_reviews"] = dfs[i]["review"].apply( lambda x: np.array(encode_sentence(x, vocab2index)) )
    dfs[i]["encoded_description"] = dfs[i]["product"].apply( lambda x: np.array(encode_sentence(x, vocab2index)) )
# 
X_train = list(dfs[0][['encoded_reviews', 'encoded_description']].to_records(index=False))
X_valid = list(dfs[2][['encoded_reviews', 'encoded_description']].to_records(index=False))
X_test = list(dfs[1][['encoded_reviews', 'encoded_description']].to_records(index=False))
#
y_train = list(dfs[0]['class'])
y_valid = list(dfs[2]['class'])
y_test = list(dfs[1]['class'])
# cross entropy loss.
criterion = nn.CrossEntropyLoss()

# review dataset class.
class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0][0].astype(np.int32)), self.y[idx], self.X[idx][1][0], self.X[idx][0][-1], self.X[idx][1][-1]
    
train_ds = ReviewsDataset(X_train, y_train)
valid_ds = ReviewsDataset(X_valid, y_valid)
test_ds = ReviewsDataset(X_test, y_test)

# confusion matrix class.
class ConfusionMatrix:
    def __init__(self, num_classes: int=2):
        self.num_classes = num_classes
        self.reset()
    
    def get(self):
        return torch.as_tensor(self.unnorm_conf_mat).int().numpy()
    
    def show(self, path: Union[str, None]=None):
        import numpy as np
        import matplotlib.pyplot as plt

        data = self.get()
        fig, ax = plt.subplots(dpi=150)
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(data, cmap="Blues")
        for (i, j), z in np.ndenumerate(data):
            ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center', fontsize=7)
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(
            ["Related", "Same Brand\nand Category", 
             "Same Brand\nDifferent Category", 
             "Differet Brand\nSame Category", 
             "Different Brand\nand Category"], 
            rotation=90, fontsize=7
        )
        ax.set_yticks([0,1,2,3,4])
        ax.set_yticklabels(
            ["Related", "Same Brand\nand Category", 
             "Same Brand\nDifferent Category", 
             "Differet Brand\nSame Category", 
             "Different Brand\nand Category"], 
                           rotation=0, fontsize=7)
        ax.set_xlabel("Predictions", fontsize=9, fontweight="bold")
        ax.set_ylabel("True Values", fontsize=9, fontweight="bold")
        plt.tight_layout()
        if path: 
            print(f"\x1b[34;1msaving confusion matrix at: \x1b[0m{path}")
            plt.savefig(path)
        else: plt.show()
    
    def reset(self):        
        num_classes =  self.num_classes
        self.unnorm_conf_mat = np.zeros((num_classes, num_classes)) 
        # torch.eye(self.num_classes)
    def update(self, logits, labels):
        self.unnorm_conf_mat += confusion_matrix(
            labels, logits, 
            labels=list(
                range(
                    self.num_classes
                )
            )
        )

def train_model(model, epochs=10, lr=0.001, device="cpu", save_path="twin_lstm.pt"):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    best_val_acc = 0
    best_epoch = 0
    # best_model = None
    for i in range(epochs):
        model.train()
        sum_loss, total = 0, 0

        train_bar = tqdm(enumerate(train_dl), total = len(train_dl), desc = "")
        for step, (x, y, l, x_mask, l_mask) in train_bar:
            # move all tensors to target device.
            l = l.to(device)
            x = x.long().to(device)
            y = y.long().to(device)
            x_mask = x_mask.to(device)
            l_mask = l_mask.to(device)
            
            y_pred = model(x, l, x_mask, l_mask)
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
            train_bar.set_description(f"train : epoch: {i + 1}/{epochs} loss : {loss.item():.2f}")
        val_loss, val_acc, val_conf_matrix, val_class_acc, val_f1 = validation_metrics(model, val_dl, i)
        # print(f"type(val_loss) = {type(val_loss)}")
        if val_acc > best_val_acc:
            best_epoch = i
            # best_model = model 
            best_val_acc = val_acc  
            torch.save(model.state_dict(), save_path)
            print(f"\x1b[32;1msaving best model with val_acc={val_acc} at {save_path}\x1b[0m")
        # val_conf_matrix.show("LSTM_confusion_matrix_.svg")
        # if True:
        print("train loss %.3f, val loss %.3f, val accuracy %.3f" % (sum_loss/total, val_loss, val_acc))
        if i-best_epoch>5:
            print("\x1b[31;1mtriggered early stopping\x1b[0m")
            break
    metrics = {
        "acc": val_acc,
        "loss": val_loss,
        "f1_score": val_f1,
        "class_acc": val_class_acc,
    }
    print(metrics)
    print(f"best val acc: {100*best_val_acc:.3f}%")
    
    return model
        
def validation_metrics (model, valid_dl, epoch):
    model.eval()
    correct, total, sum_loss = 0, 0, 0
    all_pred, all_y = [], []
    
    for x, y, l, x_mask, l_mask in tqdm(valid_dl):
        x = x.long().to(device)
        y = y.long().to(device)
        l = l.long().to(device)
        x_mask = x_mask.to(device)
        l_mask = l_mask.to(device)
        
        with torch.no_grad():
            y_hat = model(x, l, x_mask, l_mask)
            loss = criterion(y_hat, y)
            pred = torch.max(y_hat, 1)[1]
            correct += (pred == y).float().sum()
            all_y.extend(list( np.array(y.cpu()) ))
            all_pred.extend(list( np.array(pred.cpu()) ))
            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]
            # sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    print("Class wise accuracies : ")
    all_y = np.array(all_y)
    all_pred = np.array(all_pred)
    val_f1 = f1_score(all_y, all_pred, average="macro")
    val_class_acc = []
    for i in range(5):
        print("Class : ", i , ": ", (np.logical_and((all_y == i), (all_pred == i))).sum()/((all_y == i).sum()))
        val_class_acc.append((
            np.logical_and(
                (all_y == i), 
                (all_pred == i)
            )
        ).sum()/(
            (all_y == i).sum()
        ))
    val_conf_matrix = ConfusionMatrix(5)
    val_conf_matrix.reset()
    val_conf_matrix.update(all_pred, all_y)
    # print(f"type(val_class_acc) = {type(val_class_acc)}")
    # val_conf_matrix.show("LSTM_confusion_matrix_" + str(epoch) + ".svg")
    return sum_loss/total, correct/total, val_conf_matrix, val_class_acc, val_f1

# # move to argument parser.
# batch_size = 500
# vocab_size = len(words)
# # create dataloaders.
# train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_dl = DataLoader(valid_ds, batch_size=batch_size)

def load_glove_vectors(glove_file="glove.6B.100d.txt"):
    """Load the glove word vectors"""
    word_vectors = {}
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    return word_vectors

def get_emb_matrix(pretrained, word_counts, emb_size = 100):
    """ Creates embedding matrix from word vectors"""
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in word_vecs:
            W[i] = word_vecs[word]
        else:
            W[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1   
    return W, np.array(vocab), vocab_to_idx

# # move to argument parser.
# batch_size = 500
# vocab_size = len(words)
# # create dataloaders.
# train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_dl = DataLoader(valid_ds, batch_size=batch_size)
# # load word vectors (GloVe).
# word_vecs = load_glove_vectors()
# pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, counts)

class LSTM_glove_vecs(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        self.embeddings.weight.requires_grad = False ## freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # heuristics MLP.
        self.linear = nn.Linear(4*hidden_dim, 5)
        # self.softmax = nn.Softmax(dim = 0)
        # self.dropout = nn.Dropout(0.2)
        
    def pool_sequence_embedding(self, 
                                seq_token_embeds: torch.Tensor,
                                seq_token_masks: torch.Tensor) -> torch.Tensor:
        D = seq_token_embeds.shape[-1]
        # batch_size x seq_len -> batch_size x seq_len x embed_dim
        seq_token_masks = seq_token_masks.unsqueeze(dim=-1).repeat(1,1,D) 
        return (seq_token_embeds * seq_token_masks).sum(1)/(seq_token_masks.sum(1) + 1e-5)
    
    def forward(self, x, l, x_mask, l_mask):
        x = self.embeddings(x)
        # x = self.dropout(x)
        l = self.embeddings(l)
        # h0 = torch.randn(1, 500, 100)
        # c0 = torch.randn(1, 500, 100)
        lstm_out, (ht, ct) = self.lstm(x)

        lstm_out1, (ht1, ct1) = self.lstm(l)
        masked_pooled_out = self.pool_sequence_embedding(lstm_out, x_mask)
        masked_pooled_out1 = self.pool_sequence_embedding(lstm_out1, l_mask)
        
        y_hat = self.linear(torch.cat([masked_pooled_out, masked_pooled_out1, masked_pooled_out-masked_pooled_out, masked_pooled_out*masked_pooled_out1], 1))
        # y_hat = self.softmax(y_hat)
        return y_hat

# def get_args():
#     parser = argparse.ArgumentParser("")
#     parser.add_argument("")
    
#     return parser.parse_args()
    
    
if __name__ == "__main__":
    # args = get_args() # get argumens.
    args = argparse.Namespace()
    args.device = "cuda:1"
    args.batch_size = 128
    # to be incorporated in argparse.
    device = args.device
    batch_size = args.batch_size
    vocab_size = len(words)
    print(args)
    # create dataloaders.
    train_dl = DataLoader(train_ds, shuffle=True, 
                          batch_size=batch_size)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)
    # load word vectors (GloVe).
    word_vecs = load_glove_vectors()
    pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, word_set)
    
    model = LSTM_glove_vecs(vocab_size, 100, 100, 
                            pretrained_weights)
    model.to(device)
    
    exp_folder = os.path.join("experiments", "TwinLSTM")
    os.makedirs(exp_folder, exist_ok=True)
    save_path = os.path.join(exp_folder, "model.pt")
    best_model = train_model(model, epochs=30, 
                             lr=0.01, device=device,
                             save_path=save_path)
    best_state_dict = torch.load(save_path)
    best_model.load_state_dict(best_state_dict)
    print("verifying model is the correct best model by re-running validation")
    dl = DataLoader(valid_ds, batch_size=batch_size)
    val_loss, val_acc, val_conf_matrix, val_class_acc, _ = validation_metrics(best_model, dl, 0)
    print(f"val_acc: {100*val_acc:.3f}%")
    
    print("running on test set")
    dl = DataLoader(test_ds, batch_size=batch_size)
    test_loss, test_acc, test_conf_matrix, test_class_acc, test_f1 = validation_metrics(best_model, dl, 0)
    print(f"test_f1: {100*test_f1:.3f}")
    print(f"test_acc: {100*test_acc:.3f}")
    print(f"test_class_acc: {test_class_acc}")
    metrics_path = os.path.join(exp_folder, "test_metrics.json")
    metrics = {
        "acc": test_acc,
        "loss": test_loss,
        "f1_score": test_f1,
        "class_acc": test_class_acc,
    }
    conf_mat_path = os.path.join(exp_folder, "confusion_matrix.png")
    test_conf_matrix.show(conf_mat_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    with open(matrix_path, "w") as f:
        json.dump(test_conf_matrix.tolist(), f)