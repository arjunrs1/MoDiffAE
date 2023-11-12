#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
#import scipy.stats as stats
#import seaborn as sns
import matplotlib.lines as lines
import numpy as np
import os
import pandas as pd
from io import StringIO
from collections import Counter
from utils.karate import data_info

cwd = os.getcwd()
datapath = os.path.join(cwd, 'datasets', 'karate', 'karate_motion_modified.npy')
data = np.load(datapath, allow_pickle=True)

#######################

labels = ['9 kyu', '8 kyu', '7 kyu', '6 kyu', '5 kyu', '4 kyu', '3 kyu', '2 kyu', '1 kyu', '1 dan', '2 dan', '3 dan', '4 dan']

sums = {}
for technique_cls in data_info.technique_class_to_name.keys():
    technique_sums = []
    for l in labels:
        s = len([d for d in data if d['grade'] == l and d['technique_cls'] == technique_cls])
        technique_sums.append(s)
    sums[technique_cls] = technique_sums

x = np.arange(len(labels))
width = 0.15

fs = 11

fig, ax = plt.subplots()
fig.set_size_inches(12, 5)
position = x - 2 * width

rects_sum_dict = {}
rects_list = []
for technique_cls, values in sums.items():
    technique = data_info.technique_class_to_name[technique_cls]
    rects = ax.bar(position, values, width, label=technique, alpha=0.75, edgecolor='black')
    rects_list.append(rects)
    s = len([d for d in data if d['grade'] == l and d['technique_cls'] == technique_cls]) 
    position += width

ax.set_ylabel('Number of samples', fontsize=fs+2)
ax.set_xlabel('Grade', fontsize=fs+2)
ax.set_xticks(x)
ax.set_xticklabels(labels)

#fig.yticks(fontsize=fs)
ax.tick_params(labelsize=fs+2)

ax.legend(fontsize=fs)


fig.tight_layout()

ax.set_ylim(0, 150)

for i, l in enumerate(labels):
    max_y = max([r[i].get_height() for r in rects_list])
    
    middle_x = rects_list[2][i].get_x() + (width / 2)
    #print(middle_x)

    count = len([d for d in data if d['grade'] == l]) 

    nr_participants = len(list(set([d['attacker_code'] for d in data if d['grade'] == l]))) 
    
    ax.annotate(f'{count}/{nr_participants}', xy=(middle_x, max_y + 2), xytext=(0, 15), textcoords="offset points", fontsize=fs,
                ha='center', va='bottom',
                bbox=dict(boxstyle='square', fc='white', color='black'), #color='dimgray'),
                arrowprops=dict(arrowstyle='-[, widthB=2.25, lengthB=1.25', lw=1.0, color='black')) #color='dimgray'))



plt.show()

###################






###################


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#autolabel(rects1)
#autolabel(rects2)