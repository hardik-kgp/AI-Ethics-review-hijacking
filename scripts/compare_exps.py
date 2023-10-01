#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# code to compare experiments.
import os
import sys
import json
import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt
# custom color map moving from red (low) to green (high)

colors = ["darkred", "red", "lightcoral", "white", 
          "palegreen", "green", "darkgreen"]
values = [0, 0.15, 0.4, 0.5, 0.6, 0.9, 1.0]
l = list(zip(values, colors))
cmap = clr.LinearSegmentedColormap.from_list('rg', l, N=256)

conf_mats = []
for i in range(2):
    conf_mats.append(
        np.array(
            json.load(
                open(
                    os.path.join(
                        "experiments", 
                        sys.argv[i+1], 
                        "test_conf_mat.json"
                    )
                )
            )
        )
    )
# diagonal flipping matrix.
N = len(conf_mats[0])
diag_flip_mat = np.ones(N)-2*np.eye(N)
# assuming first index (0) is emanual and second (1) is without emanual.
annot_mat = conf_mats[0]-conf_mats[1]
diff_mat = (conf_mats[1]-conf_mats[0])*diag_flip_mat # flip back the diagonal terms as they denote the TPs which should inc.
diff_name = f"{sys.argv[1]}_{sys.argv[2]}_diff.png"
fig, ax = plt.subplots(dpi=150)
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(diff_mat, cmap=cmap)
for (i, j), z in np.ndenumerate(annot_mat):
    ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center', fontsize=7)
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(
    ["related", "sbsc", 
     "sbdc", "dbsc", "dbdc"], 
    rotation=90, fontsize=7
)
ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels(
    ["related", "sbsc", 
     "sbdc", "dbsc", "dbdc"], 
    rotation=0, fontsize=7
)
ax.set_xlabel("Predictions", fontsize=9, fontweight="bold")
ax.set_ylabel("True Values", fontsize=9, fontweight="bold")
plt.tight_layout()
print(f"\x1b[34;1msaving performance difference matrix at: \x1b[0m{diff_name}")
plt.savefig(diff_name)