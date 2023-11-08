#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import matplotlib.lines as lines
import numpy as np
import os
import pandas as pd
from io import StringIO
from collections import Counter

cwd = os.getcwd()
participant_data = open(os.path.join(cwd, '..', 'participants.csv'), 'r').readlines()
# Deleting the unit row. 
del participant_data[1]
participants_df = pd.read_csv(StringIO(','.join(participant_data)), sep=',', header=0)
participants_df.set_index('Code', drop=True, inplace=True)

ages = participants_df['Age'].values
weights = participants_df['Weight'].values
heights = participants_df['Height'].values
experiences = participants_df['Experience'].values
genders = participants_df['Gender'].values
grades = participants_df['Grade'].values


fig, ax = plt.subplots()
sns.histplot(data=ages, kde=True, discrete=True,  ax=ax, line_kws={'linestyle': 'dashed'})
plt.show()

fig, ax = plt.subplots()
sns.histplot(data=weights, kde=True, discrete=True,  ax=ax, line_kws={'linestyle': 'dashed'})
plt.show()

fig, ax = plt.subplots()
sns.histplot(data=heights, kde=True, discrete=True,  ax=ax, line_kws={'linestyle': 'dashed'})
plt.show()

fig, ax = plt.subplots()
sns.histplot(data=experiences, kde=True, discrete=True,  ax=ax, line_kws={'linestyle': 'dashed'})
plt.show()

# This graph is unneccesary 
gender_freqs = Counter(genders)
xvals = range(len(gender_freqs.values()))
plt.bar(xvals, gender_freqs.values() , color='tab:blue')
plt.show()
#sns.barplot(data=participants_df, x='Grade', y='Experience', order=participants_df.sort_values('Experience').Grade)

###############

fig = plt.figure()
fig.set_figwidth(7)
fig.set_figheight(4)
counts = participants_df.groupby('Grade')['Experience'].count()
labels = ['9 kyu', '8 kyu', '7 kyu', '6 kyu', '5 kyu', '4 kyu', '3 kyu', '2 kyu', '1 kyu', '1 dan', '2 dan', '3 dan', '4 dan']
ordered_counts = [counts[x] for x in labels]
ax = sns.barplot(x=labels, y=ordered_counts, color='tab:blue') #, palette=cols)
plt.show()

###########

# Relevant two plots: 

# TODO: add standard deviations. see seaborn stuff. 

fig = plt.figure()
fig.set_figwidth(7)
fig.set_figheight(4)
means = participants_df.groupby('Grade')['Experience'].mean()
stds = participants_df.groupby('Grade')['Experience'].std()
stds = stds.fillna(0)
labels = ['9 kyu', '8 kyu', '7 kyu', '6 kyu', '5 kyu', '4 kyu', '3 kyu', '2 kyu', '1 kyu', '1 dan', '2 dan', '3 dan', '4 dan']
ordered_means = [means[x] for x in labels]
ordered_stds = [stds[x] for x in labels]
ax = sns.barplot(x=labels, y=ordered_means, yerr=ordered_stds, color='tab:blue', alpha=0.75, edgecolor='black', capsize=0.4) #, palette=cols)
#ax = sns.barplot(x=labels, y=ordered_means, yerr=ordered_stds, color='tab:blue', alpha=0.75, edgecolor='black', capsize=0.4) #, palette=cols)
#ax.capsize = 0.5
#print(ax)
#exit()
ax.set_xlabel('Grade')
ax.set_ylabel('Experience in years')
plt.show()


fig = plt.figure()
fig.set_figwidth(7)
fig.set_figheight(4)
means = participants_df.groupby('Grade')['Height'].mean()
stds = participants_df.groupby('Grade')['Height'].std()
stds = stds.fillna(0)
labels = ['9 kyu', '8 kyu', '7 kyu', '6 kyu', '5 kyu', '4 kyu', '3 kyu', '2 kyu', '1 kyu', '1 dan', '2 dan', '3 dan', '4 dan']
ordered_means = [means[x] for x in labels]
ordered_stds = [stds[x] for x in labels]
ax = sns.barplot(x=labels, y=ordered_means, yerr=ordered_stds, color='tab:blue', alpha=0.75, edgecolor='black', capsize=0.4) #, palette=cols)
ax.set_ylim(140, 195)
ax.set_xlabel('Grade')
ax.set_ylabel('Height in cm')
plt.show()

print(counts)


###############

n_colors = 100 # 13
palette=sns.color_palette("ch:s=.25,rot=-.25", n_colors=n_colors)

fig = plt.figure()
fig.set_figwidth(7)
fig.set_figheight(4)
max_experience = np.max(experiences)
labels = ['9 kyu', '8 kyu', '7 kyu', '6 kyu', '5 kyu', '4 kyu', '3 kyu', '2 kyu', '1 kyu', '1 dan', '2 dan', '3 dan', '4 dan']
means = participants_df.groupby('Grade')['Experience'].mean()
label_means = [means[x] for x in labels]
cols = [palette[int(x / max_experience * n_colors) - 1] for x in label_means]
#ax = sns.barplot(data=participants_df, x='Grade', y='Experience', palette=cols)
ax = sns.barplot(x=labels, y=label_means, palette=cols)
#ax = sns.barplot(data=participants_df, x='Grade', y='Experience', palette=sns.color_palette("ch:s=.25,rot=-.25", n_colors=13))
#ax.set_xticks(range(len(np.unique(grades))))
#ax.set_xticklabels(labels)
ax.set_ylabel('Experience in years')
plt.show()





edit_df = participants_df
edit_df.drop('B0404', inplace=True)
edit_df.drop('B0405', inplace=True)
fig = plt.figure()
fig.set_figwidth(7)
fig.set_figheight(4)
max_experience = np.max(experiences)
labels = ['9 kyu', '8 kyu', '7 kyu', '6 kyu', '5 kyu', '4 kyu', '3 kyu', '2 kyu', '1 kyu', '1 dan', '3 dan', '4 dan']
means = edit_df.groupby('Grade')['Experience'].mean()
label_means = [means[x] for x in labels]
cols = [palette[int(x / max_experience * n_colors) - 1] for x in label_means]
#ax = sns.barplot(data=participants_df, x='Grade', y='Experience', palette=cols)
ax = sns.barplot(x=labels, y=label_means, palette=cols)
#ax = sns.barplot(data=participants_df, x='Grade', y='Experience', palette=sns.color_palette("ch:s=.25,rot=-.25", n_colors=13))
#ax.set_xticks(range(len(np.unique(grades))))
#ax.set_xticklabels(labels)
ax.set_ylabel('Experience in years')
plt.show()

##############

