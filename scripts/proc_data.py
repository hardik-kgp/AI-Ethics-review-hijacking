#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# convert dataset to tsv for easier use of TSVDataset class of gluonNLP
import os
import json
from tqdm import tqdm
from typing import List
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def read_jsonl(path: str) -> List[dict]:
    """read jsonl dataset."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            rec = json.loads(line)
            data.append(rec)
    
    return data

def populate_prod_info(meta_path: str, data: List[dict]) -> List[dict]:
    prod_info = read_jsonl(meta_path)
    prod_info_map = {}
    for info in prod_info:
        prod_info_map[info["asin"]] = info
    for rec in tqdm(data, desc="populating product metadata"):
        prod_rec = prod_info_map[rec["asin"]]
        rec["title"] = prod_rec["title"]
        rec["feature"] = prod_rec["feature"]
        rec["description"] = prod_rec["description"]
    
    return data

def process_features(features: list, k: int=5):
    proc_features = []
    for feature in features:
        feature = BeautifulSoup(feature, features="html.parser").text.strip()
        feature = " ".join(feature.split())
        proc_features.append(feature)
        
    return proc_features[:k]

def process_description(features: list, k: int=5):
    proc_features = []
    for feature in features:
        feature = BeautifulSoup(feature, features="html.parser").text.strip()
        feature = " ".join(feature.split())
        proc_features.append(feature)
        
    return proc_features[:k]

def process_product(prod_info, use_brand=False):
    title = prod_info["post_title"]
    feat_text = " ".join(process_features(
        prod_info["feature"]
    ))
    desc_text = " ".join(process_description(
        prod_info["description"]
    ))
    prod_text =  title + " " + feat_text + " " + desc_text
    if use_brand: prod_text = prod_info["brand"] + " " + prod_text
    prod_text = " ".join(prod_text.split()).strip()
    
    return prod_text
    
def process_review(review_info):
    rev_text = " ".join(review_info["reviewText"].split("\n")) # review text.
    rev_title = " ".join(review_info["summary"].split("\n")) # title/summary.
    review_text = rev_title + " " + rev_text
    review_text = " ".join(review_text.split()).strip()
    
    return review_text
    
def convert_to_tsv_and_split(path: str, meta_path: str="meta_Appliances.json", 
                             save_path: str="synth_prod_review_sim",
                             random_state=42) -> None:
    import pandas as pd
    data = read_jsonl(path)
    # populate product info.
    data = populate_prod_info(meta_path, data)
    # convert to specific format.
    proc_data = []
    class_map = {'-': 0, "sbsc": 1, "sbdc": 2, "dbsc": 3, "dbdc": 4}
    for rec in tqdm(data, desc="converting to tsv"):
        new_rec = {}
        new_rec["id"] = rec["id"]
        new_rec["asin"] = rec["asin"]
        new_rec["product"] = process_product(rec)
        new_rec["review"] = process_review(rec)
        new_rec["class"] = class_map.get(rec["sub_class"])
        new_rec["related"] = rec["related"]
        new_rec["brand"] = rec["brand"]
        new_rec["subcateg"] = rec["category"][1].replace("&amp;", "&")
        proc_data.append(new_rec)
    df = pd.DataFrame(proc_data)
    X = list(range(len(df)))
    y = list(df["class"])
    train_ind_seq, test_ind_seq, _, _ = train_test_split(
        X, y, shuffle=True, 
        test_size=0.2, stratify=y,
        random_state=random_state, 
    )
    records = df.to_dict("records") 
    test_records = []
    train_records = []
    for i in train_ind_seq:
        train_records.append(records[i])
    for i in test_ind_seq:
        test_records.append(records[i])
    test_df = pd.DataFrame(test_records)
    train_df = pd.DataFrame(train_records)
    train_df.to_csv(save_path+"_train.tsv", sep="\t", index=False)
    test_df.to_csv(save_path+"_test.tsv", sep="\t", index=False)
    
    
def convert_to_tsv(path: str, random_state=42, use_brand=False) -> None:
    import pandas as pd
    data = json.load(open(path))
    # convert to specific format.
    err_ctr = 0
    proc_data = []
    class_map = {'-': 0, "sbsc": 1, "sbdc": 2, "dbsc": 3, "dbdc": 4}
    # print(type(data[0][0]), len(data[0]), data[0][0].keys())
    for rec in tqdm(data, desc="converting to tsv"):
        try:
            new_rec = {}
            new_rec["id"] = rec["id"]
            new_rec["asin"] = rec["asin"]
            new_rec["product"] = process_product(rec, use_brand=use_brand)
            new_rec["review"] = process_review(rec)
            new_rec["class"] = class_map.get(rec["sub_class"])
            new_rec["related"] = rec["related"]
            new_rec["brand"] = rec["brand"]
            new_rec["subcateg"] = rec["category"][1].replace("&amp;", "&")
            proc_data.append(new_rec)
        except KeyError as e:
            err_ctr += 1
            print(f"({err_ctr}):", e)
    print(f"{err_ctr} errors  in {path}")    
    df = pd.DataFrame(proc_data)
    path_w_ext, ext = os.path.splitext(path)
    df.to_csv(path_w_ext+".tsv", sep="\t", index=False)
        
        
if __name__ == "__main__":
    # convert_to_tsv_and_split("fake_hijacked_application_reviews.jsonl")
    use_brand = True
    convert_to_tsv(
        "balanced_review_hijack_train.json", 
        use_brand=use_brand,
    )
    convert_to_tsv(
        "balanced_review_hijack_test.json", 
        use_brand=use_brand,
    )
    convert_to_tsv(
        "balanced_review_hijack_val.json", 
        use_brand=use_brand,
    )