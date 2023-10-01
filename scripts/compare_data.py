#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BertTokenizerFast, AutoTokenizer
# default = pd.read_csv("data/balanced_review_hijack_test.tsv", sep="\t")
# emanual = pd.read_csv("data/emanual_balanced_review_hijack_test.tsv", sep="\t")
# dataset class
class ProductReviewDataset(Dataset):
    def __init__(self, path: str, tokenizer: Union[None, AutoTokenizer]=None, **tok_args):
        self.path = path
        self.data = pd.read_csv(path, sep="\t").to_dict("records")
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        
    def get_class_weights(self):
        class_weights = {}
        for rec in self.data:
            try: class_weights[rec["class"]] += 1
            except KeyError: class_weights[rec["class"]] = 1
        class_weights = [v for _,v in sorted(class_weights.items(), key=lambda x: x[0])]
                
        return torch.as_tensor(class_weights, dtype=torch.float)
        
    def __len__(self):
        return len(self.data)

    def get_product(self, i: int) -> List[str]:
        rec = self.data[i]
        product = rec["product"]
        
        return product.split()  
    
    def get_review(self, i: int) -> List[str]:
        rec = self.data[i]
        review = rec["review"]
        
        return review.split()        
    
    def get_tokenized_review(self, i: int) -> torch.Tensor:
        rec = self.data[i]
        review = rec["review"]
        tokenized_review = self.tokenizer(review, return_tensors="pt")
        tokenized_review = tokenized_review["input_ids"][0][1:-1]
        # print(tokenized_review)
        return tokenized_review

    def __getitem__(self, i: int):
        rec = self.data[i]
        s1 = rec["product"]
        s2 = rec["review"]
        label = rec["class"]
        enc_dict = self.tokenizer(s1, s2, **self.tok_args)
        try:
            return [
                enc_dict["input_ids"][0], 
                enc_dict["attention_mask"][0], 
                enc_dict["token_type_ids"][0], 
                label
            ]
        except:
            return [
                enc_dict["input_ids"][0], 
                enc_dict["attention_mask"][0],
                label
            ]

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
d_path = "data/balanced_review_hijack_test.tsv"
e_path = "data/emanual_balanced_review_hijack_test.tsv"
default = ProductReviewDataset(
            d_path, tokenizer=tokenizer,
            max_length=250, padding="max_length",
            truncation=True, return_tensors="pt",
        )
emanual = ProductReviewDataset(
            e_path, tokenizer=tokenizer,
            max_length=250, padding="max_length",
            truncation=True, return_tensors="pt",
        )
overlaps = []
jaccards = [] 
# prod_overlaps = []
prod_jaccards = []
pbar = tqdm(zip(default, emanual), total=len(default), 
            desc="computing overlap in token sequences")
for d, e in pbar:
    dids = d[0].tolist()
    eids = e[0].tolist()
    
    tot = len(dids)
    matches = 0
    assert len(eids) == tot, "inconsistent padding/truncation"
    for i,j in zip(dids, eids):
        matches += int(i == j) 
    set_d = set(dids)
    set_e = set(eids)
    union = len(set_d.union(set_e))
    intersection = len(set_d.intersection(set_e))
    overlaps.append(matches/tot)
    jaccards.append(intersection/union)
    
    tot = 0
    matches = 0
    dids = dids[:dids.index(102)]
    eids = eids[:eids.index(102)]
    union = len(set_d.union(set_e))
    set_d = set(dids)
    set_e = set(eids)
    intersection = len(set_d.intersection(set_e))
#     for i,j in zip(dids, eids):
#         matches += int(i == j) 
#         tot += 1
#     prod_overlaps.append(matches/tot)
    prod_jaccards.append(intersection/union)
    
print(f"overlap[total] = {100*np.mean(overlaps):.3f} ± {100*(np.var(overlaps)**0.5):.3f}")
print(f"jaccard[total] = {100*np.mean(jaccards):.3f} ± {100*(np.var(jaccards)**0.5):.3f}")
# print(f"overlap[prod] = {100*np.mean(prod_overlaps):.3f} ± {100*(np.var(prod_overlaps)**0.5):.3f}")
print(f"jaccard[prod] = {100*np.mean(prod_jaccards):.3f} ± {100*(np.var(prod_jaccards)**0.5):.3f}")