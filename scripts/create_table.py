#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# code to compare experiments.
import os
import sys
import json
import numpy as np
from typing import *
# import matplotlib.colors as clr
# import matplotlib.pyplot as plt

try:
    exp_folder = sys.argv[1]
except IndexError:
    exp_folder = "./experiments"
    print("no experiments folder path given. Defaulting to \x1b[34;1m`./experiments`\x1b[0m")
    if not os.path.exists("experiments"):
        exit("\x1b[31;1mDidn't find `./experiments` folder. Terminating :(\x1b[0m")
    
class Table:
    def __init__(self, *fields):
        self.fields = fields
        self.values = []
        self.highlighted_cells = []
        
    def __len__(self):
        return len(self.fields)
        
    def append(self, row: list):
        assert len(self) == len(row), f"mismatched values in row ({len(row)}), compared to {len(self)} fields"
        self.values.append([str(i) for i in row])
        
    def sort(self, by: int=0, reverse=False):
        """ sort by column index `by` """
        self.values = sorted(self.values, key=lambda x: x[by], reverse=reverse)
        
    def find_max_in_col(self, col_id: int=0):
        try:
            return np.argmax([float(row[col_id]) for row in self.values])
        except Exception as e:
            print(f"\x1b[31;1m{e}. failing silently: first row of column {col_id} will be highlighted.\x1b[0m")
            return 0
    
    def find_max_in_cols(self, col_ids: List[int]):
        row_ids = []
        for i in col_ids:
            row_ids.append(self.find_max_in_col(i))
        
        return row_ids
    
    def highlight_max(self, col_ids: List[int]):
        self.highlighted_cells = []
        row_ids = self.find_max_in_cols(col_ids)
        for i,j in zip(row_ids, col_ids):
            self.highlighted_cells.append((i, j))
    
    def __str__(self):
        op = "|"+"|".join(self.fields)+"|\n"+"|"+"|".join(["---"]*len(self))+"|"
        for i, row in enumerate(self.values):
            row = [f"**{val}**" if ((i,j) in self.highlighted_cells) else val for j,val in enumerate(row)]
            op += "\n|"+"|".join(row)+"|"
            
        return op
    
    
def get_name(file):
    argv = file.split("_")
    if len(argv) == 1:
        return file
    model = argv[1]
    emanual_str = ""
    if argv[0] == "EManual":
        emanual_str += "e-manual"
        if argv[-1] == "rb":
            emanual_str += " review (back)"
        elif argv[-1] == "prb":
            emanual_str += " product+review (back)"
        else:
            emanual_str += " product (back)"
    # print(model, emanual_str)
    return (model + f" {emanual_str}").strip()
# skip_list = ["EManualBERT", "ReviewBERT", ""]
skip_list = ["EManual_BERT", "EManual_RoBERTa"]
# table = """|model|acc|macro f1|related acc|sbsc acc|sbdc acc|dbsc acc|dbdc acc|
# |---|---|---|---|---|---|---|---|"""
table = Table("model", "acc", "macro f1", 
              "related acc", "sbsc acc", 
              "sbdc acc", "dbsc acc", "dbdc acc")
for file in os.listdir(exp_folder):
    if file in skip_list: continue
    path = os.path.join(exp_folder, file, "test_metrics.json")
    if not os.path.exists(path): 
        print(f"no test metrics found for \x1b[34;1m{file}\x1b[0m")
        continue
    # print(get_name(file))
    try:
        metrics = json.load(open(path))
    except json.decoder.JSONDecodeError as e:
        print(path, e)
    try:
        acc = round(100*metrics["acc"], 3)
        f1 = round(100*metrics["f1_score"], 3)  # macro f1 score
        class_acc = [round(100*i, 3) for i in metrics["class_acc"]]
        row = [get_name(file), acc, f1]+class_acc
        table.append(row)
        # row = f"|{get_name(file)}|{acc}|{f1}|"
        # for acc in class_acc:
        #     row += f"{acc}|"
        # table += f"\n{row}"
    except KeyError as e:
        print(f"\x1b[1mskipping \x1b[34;1m{file}\x1b[0m\x1b[1m due to missing metrics\x1b[0m ({e})")
table.sort(by=0)
table.highlight_max(range(1, len(table)))
print(table)