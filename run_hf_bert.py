#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import warnings
# warnings.filterwarnings('ignore')
import os
import json
import math
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from collections import Counter
import matplotlib.pyplot as plt
from typing import Union, Dict, List
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BertModel, RobertaModel, BertTokenizerFast, AutoTokenizer, BertTokenizer, RobertaTokenizer

# set logging level of transformers.
import transformers
transformers.logging.set_verbosity_error()

# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# dataset class
class ProductReviewPredictDataset(Dataset):
    def __init__(self, path: str, tokenizer: Union[None, AutoTokenizer]=None, **tok_args):
        self.path = path
        self.data = json.load(open(path)) 
        # pd.read_csv(path, sep="\t").to_dict("records")
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        rec = self.data[i]
        s1 = rec["product"]
        s2 = rec["review"]
        enc_dict = self.tokenizer(s1, s2, **self.tok_args)
        try:
            return [
                enc_dict["input_ids"][0], 
                enc_dict["attention_mask"][0], 
                enc_dict["token_type_ids"][0], 
            ]
        except:
            return [
                enc_dict["input_ids"][0], 
                enc_dict["attention_mask"][0],
            ]
        
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

# dataset class
class ProductReviewWindowedPredictDataset(Dataset):
    def __init__(self, path: str, tokenizer: Union[None, AutoTokenizer]=None, 
                 win_p=False, win_p_size=100, win_r=False, win_r_size=100, 
                 overlap=0.2, **tok_args):
        super(ProductReviewWindowedPredictDataset, self).__init__()
        self.path = path
        self.data = pd.read_csv(path, sep="\t").to_dict("records")
        new_data = []
        print(f"win_p: {win_p}")
        print(f"win_r: {win_r}")
        if win_p:
            print(f"applying windowed transformer on product with window size: {win_p_size}, overlap: {100*overlap:.0f}%")
        if win_r:
            print(f"applying windowed transformer on review with window size: {win_r_size}, overlap: {100*overlap:.0f}%")
        # windowed transformer approach.
        win_id = 0
        for rec in self.data:
            # the window and product chunks default to whole product/review if win_p and win_r are false respectively.
            windowed_r = [rec["review"]]
            windowed_p = [rec["product"]]
            
            if win_p:
                windowed_p = []
                chunk_size = win_p_size
                # split raw (untokenized) product info by space.
                words = rec["product"].split()
                N_p = math.ceil((len(words) - win_p_size*overlap)/(win_p_size*(1-overlap)))
                # created windowed product chunks.
                for i in range(N_p):
                    start = int(i*win_p_size*(1-overlap))
                    end = math.ceil(i*win_p_size*(1-overlap)+win_p_size)
                    windowed_p.append(" ".join(words[start : end]))
            
            if win_r:
                windowed_r = []
                chunk_size = win_r_size
                # split raw (untokenized) review by space.
                words = rec["review"].split()
                N_r = math.ceil((len(words) - win_r_size*overlap)/(win_r_size*(1-overlap)))
                # created windowed review chunks.
                for i in range(N_r):
                    start = int(i*win_r_size*(1-overlap))
                    end = math.ceil(i*win_r_size*(1-overlap)+win_r_size)
                    windowed_r.append(" ".join(words[start : end]))
            # create all chunk pairs.
            for windowed_r_i in windowed_r:
                for windowed_p_i in windowed_p:
                    new_rec = {} # new record (otherwise rec might get overwritten which can lead to issues.)
                    new_rec.update(rec)
                    new_rec["product"] = windowed_p_i
                    new_rec["review"] = windowed_r_i
                    new_rec["win_id"] = win_id
                    new_data.append(new_rec)
            win_id += 1
        # check the blowup in the data size.
        print(f"data size increased by {len(new_data)/len(self.data)} ({len(self.data)} to {len(new_data)})")
        self.data = new_data
        self.tokenizer = tokenizer
        self.tok_args = tok_args
        
    def get_win_id(self, i: int):
        return self.data[i]["win_id"]
        
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
        
# dataset class
class ProductReviewWindowedDataset(Dataset):
    def __init__(self, path: str, tokenizer: Union[None, AutoTokenizer]=None, 
                 win_p=False, win_p_size=100, win_r=False, win_r_size=100, 
                 overlap=0.2, **tok_args):
        self.path = path
        self.data = pd.read_csv(path, sep="\t").to_dict("records")
        new_data = []
        print(f"win_p: {win_p}")
        print(f"win_r: {win_r}")
        if win_p:
            print(f"applying windowed transformer on product with window size: {win_p_size}, overlap: {100*overlap:.0f}%")
        if win_r:
            print(f"applying windowed transformer on review with window size: {win_r_size}, overlap: {100*overlap:.0f}%")
        # windowed transformer approach.
        for rec in self.data:
            # the window and product chunks default to whole product/review if win_p and win_r are false respectively.
            windowed_r = [rec["review"]]
            windowed_p = [rec["product"]]
            if win_p:
                windowed_p = []
                chunk_size = win_p_size
                # split raw (untokenized) product info by space.
                words = rec["product"].split()
                N_p = math.ceil((len(words) - win_p_size*overlap)/(win_p_size*(1-overlap)))
                # created windowed product chunks.
                for i in range(N_p):
                    start = int(i*win_p_size*(1-overlap))
                    end = math.ceil(i*win_p_size*(1-overlap)+win_p_size)
                    windowed_p.append(" ".join(words[start : end]))
            
            if win_r:
                windowed_r = []
                chunk_size = win_r_size
                # split raw (untokenized) review by space.
                words = rec["review"].split()
                N_r = math.ceil((len(words) - win_r_size*overlap)/(win_r_size*(1-overlap)))
                # created windowed review chunks.
                for i in range(N_r):
                    start = int(i*win_r_size*(1-overlap))
                    end = math.ceil(i*win_r_size*(1-overlap)+win_r_size)
                    windowed_r.append(" ".join(words[start : end]))
            # create all chunk pairs.
            for windowed_r_i in windowed_r:
                for windowed_p_i in windowed_p:
                    new_rec = {} # new record (otherwise rec might get overwritten which can lead to issues.)
                    new_rec.update(rec)
                    new_rec["product"] = windowed_p_i
                    new_rec["review"] = windowed_r_i
                    new_data.append(new_rec)
        # check the blowup in the data size.
        print(f"data size increased by {len(new_data)/len(self.data)} ({len(self.data)} to {len(new_data)})")
        self.data = new_data
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
        
# metrics    
class Accuracy:
    def __init__(self, num_classes: int=2):
        self.num_classes = num_classes
        self.reset()
    
    def get(self):
        return (self.class_matches/self.class_counts)
    
    def reset(self):        
        self.class_counts = 0
        self.class_matches = 0
    
    def update(self, logits, labels):
        preds = logits.argmax(axis=1)
        # print(self.class_counts, self.class_matches)
        self.class_counts += len(labels)
        self.class_matches += ((preds == labels).sum().item())
        # print(self.class_counts, self.class_matches)

class ClassAccuracy:
    def __init__(self, num_classes: int=2):
        self.num_classes = num_classes
        self.reset()
    
    def get(self):
        return (self.class_matches / self.class_counts)
    
    def reset(self):        
        self.class_counts = torch.zeros(self.num_classes)
        self.class_matches = torch.zeros(self.num_classes)
    
    def update(self, logits, labels):
        preds = logits.argmax(axis=1)
        preds = torch.eye(self.num_classes)[preds,:]
        labels = torch.eye(self.num_classes)[labels,:]
        self.class_counts += labels.sum(axis=0)
        self.class_matches += (labels*preds).sum(axis=0)
        
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
#         ax.set_xticklabels(
#             ["Related", "Same Brand\nand Category", 
#              "Same Brand\nDifferent Category", 
#              "Differet Brand\nSame Category", 
#              "Different Brand\nand Category"], 
#             rotation=90, fontsize=7
#         )
        ax.set_xticklabels(
            ["related", "sbsc", 
             "sbdc", "dbsc", 
             "dbdc"], 
            rotation=90, fontsize=7
        )
        ax.set_yticks([0,1,2,3,4])
#         ax.set_yticklabels(
#             ["Related", "Same Brand\nand Category", 
#              "Same Brand\nDifferent Category", 
#              "Differet Brand\nSame Category", 
#              "Different Brand\nand Category"], 
#             rotation=0, fontsize=7
#         )
        ax.set_yticklabels(
            ["related", "sbsc", 
             "sbdc", "dbsc", 
             "dbdc"], 
            rotation=0, fontsize=7
        )
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
        preds = logits.argmax(axis=1)
        self.unnorm_conf_mat += confusion_matrix(
            labels, preds, 
            labels=list(
                range(
                    self.num_classes
                )
            )
        )
        # preds = torch.eye(self.num_classes)[preds,:]
        # labels = torch.eye(self.num_classes)[labels,:]
        # self.unnorm_conf_mat += (labels.T @ preds)
        # self.class_counts += labels.sum(axis=0)
        # self.class_matches += (labels*preds).sum(axis=0)
class ReviewHijackBERT(nn.Module):
    def __init__(self, model_path: str, tok_path: str, 
                 embed_size: int=768, num_classes: int=5):
        super(ReviewHijackBERT, self).__init__()
        try: self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        except:
            if "roberta" in tok_path.lower():
                self.tokenizer = RobertaTokenizer.from_pretrained(tok_path)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(tok_path)
        # BertTokenizerFast.from_pretrained(tok_path)
        if model_path.startswith("roberta"):
            self.model = RobertaModel.from_pretrained(model_path)
        else:
            self.model = BertModel.from_pretrained(model_path)
        self.loss_fn = nn.CrossEntropyLoss()
        self.mlp = nn.Linear(embed_size, num_classes)
        self.config = {
            "tok_path": tok_path,
            "embed_size": embed_size,
            "model_path": model_path,
            "num_classes": num_classes,
        }
    
    def forward(self, *bert_args):
        pooler_output = self.model(*bert_args).pooler_output
        return self.mlp(pooler_output)
        # cls_embed = self.model(*bert_args).last_hidden_state[:,0,:] # [CLS] embed pooler output.
        # return self.mlp(cls_embed)
    def test_windowed(self, test_path: str, **args):            
        device_id = args.get("device_id", "cuda:0")
        class_weights = args.get("class_weights")
        batch_size = args.get("batch_size", 32)
        exp_name = args.get("exp_name", "")
        device = torch.device(device_id)
        # windowed transformer for product & review.
        win_p: bool = args.get("win_p", False)
        win_r: bool = args.get("win_r", False)
        win_p_size: int = 150
        win_r_size: int = 150
        overlap: float = 0.2
            
        print("win_p:", win_p)
        print("win_r:", win_r)
            
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        # create test dataset a-nd dataloader.
        # we need to replace this ProductReviewPredictDataset that does mode based prediction over all candidate windows.
        testset = ProductReviewWindowedPredictDataset(
            test_path, tokenizer=self.tokenizer,
            max_length=250, padding="max_length",
            truncation=True, return_tensors="pt",
            win_p=win_p, win_r=win_r, overlap=overlap,
            win_p_size=win_p_size, win_r_size=win_r_size,
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        # move model to device.
        self.to(device)
        test_acc, test_loss, test_class_acc, test_conf_mat, test_trues, test_preds = self.val(testloader, 0, 0, device)
#         print(f"test_acc: {100*test_acc:.3f}%")
#         print(f"test_loss: {test_loss:.3f}")
#         print(f"test_class_acc: {test_class_acc}")
#         print(f"test_conf_mat:")
#         print(test_conf_mat.get())
#         test_conf_mat.show(os.path.join(
#             exp_name, 
#             "confusion_matrix.png"
#         ))
        desc = "getting final prediction by mode (over win ids)"
        pbar = tqdm(range(len(testset)), desc=desc)
        win_preds = {}
        win_trues = {}
        # function to get mode.
        def get_mode(l: list):
            return Counter(l).most_common(1)[0][0]
        
        for i in pbar:
            win_id = testset.get_win_id(i)
            try:
                win_preds[win_id].append(test_preds[i])
            except KeyError:
                win_preds[win_id] = [test_preds[i]]
            win_trues[win_id] = test_trues[i]
        # get mode prdictions.
        for k,v in win_preds.items():
            win_preds[k] = get_mode(v)
        # get metrics for true values and predictions.
        trues = list(win_preds.values())
        preds = list(win_trues.values())
        acc, class_acc, f1 = compute_metrics(trues, preds)
        metrics_path = os.path.join(exp_name, "test_metrics.json")
        # test_conf_mat_path = os.path.join(exp_name, "test_conf_mat.json")
        test_metrics = {
            "acc": acc,
            "f1_score": f1,
            "class_acc": class_acc.tolist(),
        }
        class_acc = [f"{100*i:.3f}%" for i in class_acc]
        print(f"test_acc: {100*acc:.3f}%")
        print(f"test_f1: {100*f1:.3f}%")
        print(f"test_class_acc: {class_acc}")
        # with open(test_conf_mat_path, "w") as f:
        #    json.dump(test_conf_mat.get().tolist(), f, indent=4)
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
        
    def test(self, test_path: str, **args):            
        batch_size = args.get("batch_size", 32)
        device_id = args.get("device_id", "cuda:0")
        exp_name = args.get("exp_name", "")
        device = torch.device(device_id)
        class_weights = args.get("class_weights")

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        # create test dataset a-nd dataloader.
        # we need to replace this ProductReviewPredictDataset that does mode based prediction over all candidate windows.
        testset = ProductReviewDataset(
            test_path, tokenizer=self.tokenizer,
            max_length=250, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        # move model to device.
        self.to(device)
        test_acc, test_loss, test_class_acc, test_conf_mat, test_trues, test_preds = self.val(testloader, 0, 0, device)
        print(f"test_acc: {100*test_acc:.3f}%")
        print(f"test_loss: {test_loss:.3f}")
        print(f"test_class_acc: {test_class_acc}")
        print(f"test_conf_mat:")
        print(test_conf_mat.get())
        test_conf_mat.show(os.path.join(
            exp_name, 
            "confusion_matrix.png"
        ))
        # print(test_preds[:20], test_trues[:20])
        desc = "calculating avg. review length for classified vs misclassified"
        pbar = tqdm(range(len(testset)), desc=desc)
        
        mis_prod_lengths = []
        corr_prod_lengths = []
        mis_review_lengths = []
        corr_review_lengths = []
        for i in pbar:
            token_len = len(testset.get_review(i))
            p_token_len = len(testset.get_product(i))
            if test_trues[i] == test_preds[i]:
                corr_review_lengths.append(token_len)
                corr_prod_lengths.append(p_token_len)
            else:
                mis_review_lengths.append(token_len)
                mis_prod_lengths.append(p_token_len)
        
        corr_review_lengths = np.array(corr_review_lengths)
        mis_review_lengths = np.array(mis_review_lengths)
        corr_prod_lengths = np.array(corr_prod_lengths)
        mis_prod_lengths = np.array(mis_prod_lengths)
        
        avg_corr_review_len = corr_review_lengths.mean()
        avg_mis_review_len = mis_review_lengths.mean()
        avg_corr_prod_len = corr_prod_lengths.mean()
        avg_mis_prod_len = mis_prod_lengths.mean() 
        
        metrics_path = os.path.join(exp_name, "test_metrics.json")
        f1_score_ = f1_score(test_trues, test_preds, average="macro")
        test_conf_mat_path = os.path.join(exp_name, "test_conf_mat.json")
        
        test_metrics = {
            "acc": test_acc,
            "f1_score": f1_score_,
            "class_acc": test_class_acc.tolist(),
            "avg_mis_review_len": avg_mis_review_len,
            "avg_corr_review_len": avg_corr_review_len,
        }
        
        print(f"test_f1_score: {f1_score_:.3f}")
        print(f"test_avg_mis_prod_len: {avg_mis_prod_len:.3f} ± {(np.var(mis_prod_lengths)**0.5):.3f}")
        print(f"test_avg_corr_prod_len: {avg_corr_prod_len:.3f} ± {(np.var(corr_prod_lengths)**0.5):.3f}")
        print(f"test_avg_mis_review_len: {avg_mis_review_len:.3f} ± {(np.var(mis_review_lengths)**0.5):.3f}")
        print(f"test_avg_corr_review_len: {avg_corr_review_len:.3f} ± {(np.var(corr_review_lengths)**0.5):.3f}")
        
        with open(test_conf_mat_path, "w") as f:
            json.dump(test_conf_mat.get().tolist(), f, indent=4)
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
            
    def pred_sus_score(self, data_path: str, **args):            
        batch_size = args.get("batch_size", 32)
        device_id = args.get("device_id", "cuda:0")
        exp_name = args.get("exp_name", "")
        device = torch.device(device_id)
        
        self.loss_fn = nn.CrossEntropyLoss()
        # create test dataset and dataloader.
        dataset = ProductReviewPredictDataset(
            data_path, tokenizer=self.tokenizer,
            max_length=250, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # move model to device.
        self.to(device)
        
        return self.predict(dataloader, device=device)
    
    def val(self, valloader: str, epoch_i: int=0, epochs: int=20, device="cpu"):
        self.eval()
        # val metrics.
        val_trues = []
        val_preds = []
        batch_losses = []
        val_acc = Accuracy(5)
        val_class_acc = ClassAccuracy(5)
        val_conf_mat = ConfusionMatrix(5)
        val_bar = tqdm(enumerate(valloader), total=len(valloader), 
                       desc=f"train: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
        # val step.
        for step, batch in val_bar:
            # move to device.
            if len(batch) == 4:
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
                batch[2] = batch[2].to(device)
            else:
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
            with torch.no_grad():
                # get logits.
                if len(batch) == 4:
                    logits = self(batch[0], batch[1], batch[2])
                else: logits = self(batch[0], batch[1])
                logits = logits.detach().cpu()
                # calculate loss.
                batch_loss = self.loss_fn(logits, batch[-1])    
                batch_losses.append(batch_loss.item())
                # update metrics.
                val_acc.update(logits, batch[-1])
                val_conf_mat.update(logits, batch[-1])
                val_class_acc.update(logits, batch[-1])
                val_bar.set_description(f"val: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {(np.mean(batch_losses)):.3f} acc: {100*val_acc.get():.2f}%")
                val_preds += logits.argmax(axis=1).tolist()
                val_trues += batch[-1].tolist()
            # if step == 5: break # DEBUG
        return val_acc.get(), np.mean(batch_losses), val_class_acc.get(), val_conf_mat, val_trues, val_preds
    
    def predict_logits(self, data_path: str, device: str="cpu", batch_size: int=32):
        # replace with ProductReviewWindowedPredictDataset
        dataset = ProductReviewDataset(
            data_path, tokenizer=self.tokenizer,
            max_length=250, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.eval()
        # val metrics.
        softmax = nn.Softmax(dim=-1)
        trues = []
        all_logits = []
        pbar = tqdm(enumerate(dataloader), 
                    total=len(dataloader), 
                    desc=f"predicting logits on test set")
        for step, batch in pbar:
            # move to device.
            if len(batch) == 4:
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
                batch[2] = batch[2].to(device)
            else:
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
            with torch.no_grad():
                # get logits.
                if len(batch) == 4:
                    logits = self(batch[0], batch[1], batch[2])
                else: logits = self(batch[0], batch[1])
                logits = logits.detach().cpu()
                all_logits += softmax(logits).tolist()
                trues += batch[-1].tolist()
                
        return all_logits, trues
    
    def predict(self, dataloader: DataLoader, device="cpu"):
        self.eval()
        softmax = nn.Softmax(dim=-1)
        # val metrics.
        preds = []
        pbar = tqdm(enumerate(dataloader), 
                    total=len(dataloader), 
                    desc=f"predicting sus scores")
        # val step.
        for step, batch in pbar:
            # move to device.
            if len(batch) == 3:
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
                batch[2] = batch[2].to(device)
            else:
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
            with torch.no_grad():
                # get logits.
                if len(batch) == 3:
                    logits = self(batch[0], batch[1], batch[2])
                else: logits = self(batch[0], batch[1])
                logits = softmax(logits)
                sus_scores = (1-logits[:,0]).detach().cpu()
                preds += sus_scores.tolist()
            # if step == 5: break # DEBUG
        return preds
    
    def fit(self, train_path: str, val_path: str, **args):
        lr = args.get("lr", 1e-5)
        epochs = args.get("epochs", 20)
        exp_name = args.get("exp_name", "experiments/EManualBERT")
        batch_size = args.get("batch_size", 32)
        device_id = args.get("device_id", "cuda:0")
        device = torch.device(device_id)
        print(args)
        # windowed transformer for product.
        win_p = args.get("win_p", False)
        win_p_size = 150
        # windowed transformer for review.
        win_r = args.get("win_r", False)
        win_r_size = 150
        overlap: float = 0.2
        
        self.to(device)
        self.optimizer = AdamW(self.parameters(), lr=lr, eps=1e-8)
        self.config.update({
            "epochs": epochs, 
            "exp_name": exp_name, 
            "batch_size": batch_size,
            "device": device_id,
            "win_p": win_p,
            "win_r": win_r,
            "win_p_size": win_p_size,
            "win_r_size": win_r_size,
            "overlap": overlap
        })
        self.config.update({
            "loss": str(self.loss_fn),
            "optimizer":  str(self.optimizer),
            "max_length": 250,
            "padding": "max_length",
            "truncation": True,
            "train_path": train_path,
            "val_path": val_path,
            "return_tensors": "pt",
        })
        os.makedirs(exp_name, exist_ok=True)
        config_path = os.path.join(exp_name, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        # create train and val loaders.
        trainset = ProductReviewWindowedDataset(
            train_path, tokenizer=self.tokenizer,
            win_p=win_p, win_r=win_r, 
            win_p_size=win_p_size, win_r_size=win_r_size,
            max_length=250, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        if args.get("use_weighted_loss", False):
            # set class weights for CE Loss
            class_weights = trainset.get_class_weights()
            print("class weights:", class_weights.tolist())
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else: self.loss_fn = nn.CrossEntropyLoss()
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valset = ProductReviewWindowedDataset(
            val_path, tokenizer=self.tokenizer,
            win_p=win_p, win_r=win_r, 
            win_p_size=win_p_size, win_r_size=win_r_size,
            max_length=250, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        # metrics.
        best_val_acc = 0 
        train_acc = Accuracy(5)
        train_class_acc = ClassAccuracy(5)
        # main train-val loop.
        for epoch_i in tqdm(range(epochs)):
            self.train()
            batch_losses = []
            train_acc.reset()
            train_class_acc.reset()
            train_bar = tqdm(enumerate(trainloader), total=len(trainloader), 
                             desc=f"train: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
            # train step.
            for step, batch in train_bar:
                if len(batch) == 4: # bert based model (contains `token_type_ids`)
                    # move to device.
                    batch[0] = batch[0].to(device)
                    batch[1] = batch[1].to(device)
                    batch[2] = batch[2].to(device)
                else:
                    batch[0] = batch[0].to(device)
                    batch[1] = batch[1].to(device)
                self.optimizer.zero_grad()
                if len(batch) == 4:
                    # get logits.
                    logits = self(batch[0], batch[1], batch[2])
                else: logits = self(batch[0], batch[1])
                logits = logits.cpu()
                # calculate loss.
                batch_loss = self.loss_fn(logits, batch[-1])
                # calculate gradient.
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                # accumulate batch losses.
                batch_losses.append(batch_loss.item())
                self.optimizer.step()
                train_acc.update(logits, batch[-1])
                train_class_acc.update(logits, batch[-1])
                train_bar.set_description(f"train: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {(np.mean(batch_losses)):.3f} acc: {100*train_acc.get():.2f}%")
                # if step == 5: break # DEBUG
            print(f"\x1b[32;1m train_loss: {np.mean(batch_losses):.3f} train_acc: {100*train_acc.get():.3f} train_class_acc: {train_class_acc.get()} \x1b[0m")
            # validation loop.
            val_acc, val_loss, val_class_acc, *_ = self.val(valloader, epoch_i, epochs, device)
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(exp_name, "model.pt")
                torch.save(self.state_dict(), save_path)
                print(f"saving best model with acc={val_acc} at {save_path}")
            print(f"\x1b[32;1m val_loss: {val_loss:.3f} val_acc: {100*val_acc:.3f} val_class_acc: {val_class_acc} \x1b[0m")

def compute_metrics(trues: List[int], preds: List[int]):
    class_counts = np.zeros(5)
    class_tots = np.zeros(5)
    for t, p in zip(trues, preds):
        class_counts[t] += int(t == p)
        class_tots[t] += 1
    class_acc = class_counts/class_tots
    acc = class_counts.sum()/class_tots.sum()
    f1 = f1_score(trues, preds, average="macro")
    
    return acc, class_acc, f1
            
def test_ensemble(checkpoints_list, args): 
    model_logits = []
    for exp_name, bert_path, tok_path in checkpoints_list:
        review_bert = ReviewHijackBERT(bert_path, tok_path=tok_path, 
                                       embed_size=768, num_classes=5)
        state_dict = torch.load(os.path.join("experiments", exp_name, "model.pt"), map_location="cpu")
        review_bert.load_state_dict(state_dict)
        review_bert.to(args.device)
        logits, trues = review_bert.predict_logits(data_path=args.test_path, device=args.device,
                                                   batch_size=args.batch_size)
        model_logits.append(logits)
        review_bert.to("cpu")
    total = torch.zeros(len(logits), 5)
    for model_logit in model_logits:
        total += torch.as_tensor(model_logit)
    # combined/ensembled predctions.
    preds = total.argmax(axis=1)
    acc, class_acc, f1 = compute_metrics(trues, preds)
    print(acc, class_acc, f1)
            
def finetune_bert(args):
    bert_path = args.bert_path
    exp_name = os.path.join("experiments", args.exp_name)
    tok_path = os.path.expanduser(args.tok_path)
    review_bert = ReviewHijackBERT(bert_path, tok_path=tok_path, 
                                   embed_size=768, num_classes=5)
    val_path = args.val_path # "data/balanced_review_hijack_val.tsv"
    train_path = args.train_path # "data/balanced_review_hijack_train.tsv" 
    review_bert.fit(train_path, val_path, epochs=args.epochs,
                    batch_size=args.batch_size, device_id=args.device,
                    use_weighted_loss=args.use_weighted_loss,
                    exp_name=exp_name, win_p=args.win_p, win_r=args.win_r)
              
def test_finetuned_bert(args):
    bert_path = args.bert_path
    tok_path = os.path.expanduser(args.tok_path) 
    review_bert = ReviewHijackBERT(bert_path, tok_path=tok_path, 
                                   embed_size=768, num_classes=5)
    exp_name = os.path.join("experiments", args.exp_name)
    # exp_name = "experiments/ReviewBERT_fix"
    test_path = args.test_path # "data/balanced_review_hijack_test.tsv"
    model_path = os.path.join(exp_name, "model.pt")
    state_dict = torch.load(model_path, map_location=args.device)
    # class_weights = state_dict.get("loss_fn.weight")
    try: 
        class_weights = state_dict["loss_fn.weight"]
        del state_dict["loss_fn.weight"]
    except KeyError: print("\x1b[31;1mno class weights found\x1b[0m")
    review_bert.load_state_dict(state_dict)
    print("\x1b[34;1mwin_p:\x1b[0m", args.win_p)
    print("\x1b[34;1mwin_r:\x1b[0m", args.win_r)
    if args.win_p == True or args.win_r == True: 
        review_bert.test_windowed(test_path, batch_size=args.batch_size, 
                                  device_id=args.device, exp_name=exp_name,
                                  win_r=args.win_r, win_p=args.win_p)
    else:
        review_bert.test(test_path, batch_size=args.batch_size, 
                         device_id=args.device,
                         exp_name=exp_name)

def pred_sus_score(args):
    bert_path = args.bert_path
    tok_path = os.path.expanduser(args.tok_path)
    review_bert = ReviewHijackBERT(bert_path, tok_path=tok_path, 
                                   embed_size=768, num_classes=5)
    exp_name = os.path.join("experiments", args.exp_name)
    # exp_name = "experiments/ReviewBERT_fix"
    test_path = args.test_path # "data/balanced_review_hijack_test.tsv"
    model_path = os.path.join(exp_name, "model.pt")
    state_dict = torch.load(model_path, map_location=args.device)
    # class_weights = state_dict.get("loss_fn.weight")
    try: 
        class_weights = state_dict["loss_fn.weight"]
        del state_dict["loss_fn.weight"]
    except KeyError: print("\x1b[31;1mno class weights found\x1b[0m")
    review_bert.load_state_dict(state_dict)
    sus_scores = review_bert.pred_sus_score(test_path, batch_size=args.batch_size, 
                                            device_id=args.device, exp_name=exp_name)
    pred_save_path = os.path.join(exp_name, "sus_preds.json")
    print(f"saving sus scores to \x1b[34;1m{pred_save_path}\x1b[0m")
    with open(pred_save_path, "w") as f:
        json.dump(sus_scores, f)
          
def get_args():
    parser = argparse.ArgumentParser("script to test and finetune BERT for hijacked review detection sentence pair task (given product info & review)")
    parser.add_argument("-e", "--exp_name", type=str, default="ReviewBERT_fix", help="name of the experiment (results will be saved at experiments/<exp_name>/)")
    parser.add_argument("-wr", "--win_r", action="store_true", help="apply windowed transformer approach for review.")
    parser.add_argument("-wp", "--win_p", action="store_true", help="apply windowed transformer approach for product.")
    parser.add_argument("-bp", "--bert_path", type=str, default="bert-base-uncased", help="path to pre-trained huggingface model to be used for training")
    parser.add_argument("-tkp", "--tok_path", type=str, default="~/bert-base-uncased", help="path to pre-trained/saved tokenizer")
    parser.add_argument("-t", "--train", action="store_true", help="flag to indicate finetuning. otherwise model will be loaded and tested on a test set.")
    parser.add_argument("-ps", "--pred_sus", action="store_true",  help="predict suspiciousness scores for supplied product review pairs")
    parser.add_argument("-psp", "--pred_save_path", type=str, default="case_study_prod_review_pairs.tsv", help="path to unseen case study data")
    parser.add_argument("-tep", "--test_path", type=str, default="data/balanced_review_hijack_test.tsv", help="path to unseen test data (expect tsv format)")
    parser.add_argument("-tp", "--train_path", type=str, default="data/balanced_review_hijack_train.tsv", help="path to training data (expect tsv format)")
    parser.add_argument("-vp", "--val_path", type=str, default="data/balanced_review_hijack_val.tsv", help="path to validation data (expect tsv format)")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="target device for model & data (single GPU only)")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size during training")
    parser.add_argument("-uwl", "--use_weighted_loss", action="store_true", help="use weighted loss")
    parser.add_argument("-ep", "--epochs", type=int, default=20, help="no. of epochs to run")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.train: finetune_bert(args)
    elif args.pred_sus: pred_sus_score(args)
    else: test_finetuned_bert(args)
    # test_ensemble(
    #     [
    #         ("EManualBERT_RoBERTa", "roberta-base", "roberta-base"), 
    #         ("EManualBERT_fix", "bert-base-uncased", "bert-base-uncased")
    #     ],
    #     args
    # )
#     test_ensemble(
#         [
#             ("ReviewBERT_RoBERTa", "roberta-base", "roberta-base"), 
#             ("ReviewBERT_fix", "bert-base-uncased", "bert-base-uncased")
#         ],
#         args
#     )
    # # train set distribution.
    # 20%, 17.06%, 9.06%, 24.25%, 29.62%
    # # test set distribution.
    # 20%, 17.05%, 9.07,%, 24.24%, 29.63%