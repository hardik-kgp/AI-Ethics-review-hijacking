import os
import pandas as pd

split_map = ["related", "sbsc", "sbdc", "dbsc", "dbdc"]
table = """|split|related|sbsc|sbdc|dbsc|dbdc|
|---|---|---|---|---|---|"""
for split in ["train", "val", "test"]:
    path = os.path.join("data", f"balanced_review_hijack_{split}.tsv")
    class_counts = pd.read_csv(path, sep="\t")["class"]
    class_freqs = class_counts.value_counts()
    tot = class_freqs.sum()
    class_freqs = (100*class_freqs/tot).round(3).astype(str)
    table += f"\n|{split}|{'|'.join(class_freqs)}|" 
print(table)