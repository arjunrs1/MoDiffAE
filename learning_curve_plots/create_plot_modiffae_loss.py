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

import json


cwd = os.getcwd()

base_dir = os.path.join(cwd, 'save', 'rot_6d_karate', 'modiffae_b0372', 'evaluation')
save_path = os.path.join(base_dir, 'loss')

train_scores_path = os.path.join(base_dir, 'loss_data', 'train_loss_b0372.json')
validation_scores_path = os.path.join(base_dir, 'loss_data', 'validation_loss_b0372.json')

train_loss = json.load(open(train_scores_path, 'r'))
validation_loss = json.load(open(validation_scores_path, 'r'))

overview_path = os.path.join(base_dir, 'knn_best_results_overview_b0372.json')
overview = json.load(open(overview_path, 'r'))

stopped_ckpt = int(overview['best combined checkpoint'][:-1]) * 1000

#print(len(train_loss))
#print(len(validation_loss))
#exit()

checkpoints = [str(int(ch / 1000)) + "K" for ch in [e[1] for e in train_loss]]

val_checkpoints = [str(int(ch / 1000)) + "K" for ch in [e[1] for e in validation_loss]]

print(checkpoints)
print(len(checkpoints))

'''for i in range(1001):
    if f"{i}K" not in checkpoints:
        print(f"{i}K not there")
        checkpoints.insert(i, f"{i}K")
        train_loss.insert(i, [0, i * 1000, 0.1])

for i in range(1001):
    if f"{i}K" not in val_checkpoints:
        print(f"{i}K not there")
        #checkpoints.insert(i, f"{i}K")
        validation_loss.insert(i, [0, i * 1000, 0.1])

print(checkpoints)'''

#x = checkpoints
#x[-1] = "1M"

x = []
for e in train_loss:
    #train_loss_steps.append(e[1])
    x.append(e[1])

#del checkpoints[-1]
#del train_loss[-1]
#del validation_loss[-1]

#checkpoints[-1] = "1M"

#print()
#print(len(checkpoints))
#exit()

#train_loss_steps = []
train_loss_values = []
for e in train_loss:
    #train_loss_steps.append(e[1])
    train_loss_values.append(e[2])

#validation_loss_steps = []
validation_loss_values = []
for e in validation_loss:
    #validation_loss_steps.append(e[1])
    validation_loss_values.append(e[2])

best_val_loss_step = np.argmin(validation_loss_values) * 1000
print(best_val_loss_step)

#print(type(train_loss))

plt.rc('font', size=20)

#f, ax = plt.subplots(1, 1)
f = plt.figure()
f.set_figwidth(18)
f.set_figheight(6)


plt.plot(x, train_loss_values, label="Train loss")
plt.plot(x, validation_loss_values, label="Validation loss")

'''plt.vlines(x=[stopped_ckpt], ymin=0, ymax=1, colors='black', ls='--', lw=2,
               label='Best UAR')'''

plt.vlines(x=[stopped_ckpt], ymin=plt.ylim()[0], ymax=plt.ylim()[1], colors='black', ls='--', lw=2,
               label='Chosen checkpoint')

plt.vlines(x=[best_val_loss_step], ymin=plt.ylim()[0], ymax=plt.ylim()[1], colors='black', ls=':', lw=2,
               label='Best validation loss')

plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.98))
#plt.xticks([0.0, 200000.0, 400000.0, 600000.0, 800000.0, 1000000.0])

#desired_labels = [x if int(x[:-1]) % 100 == 0 or x == '1M' else '' for i, x in enumerate(checkpoints)]


#current_x_ticks = plt.xticks()[0][1:-1]

plt.xlabel('Training steps')

desired_x_ticks = [l for l in x if l % 100000 == 0]
desired_labels = [str(int(ch / 1000)) + 'K' for ch in desired_x_ticks]
desired_labels[-1] = "1M"
'''print(desired_labels)
print(current_x_ticks)
exit()'''
#ax.set_yticklabels(desired_labels)
plt.xticks(ticks=desired_x_ticks, labels=desired_labels)

desired_y_ticks = plt.yticks()[0][1:-1]
#desired_y_ticks = [l for l in x if l % 100000 == 0]
desired_y_labels = [str(int(ch / 1000)) + 'K' for ch in desired_y_ticks]

plt.yticks(ticks=desired_y_ticks, labels=desired_y_labels)



print(list(plt.xticks()[0]))

#plt.xticks([str(x) if x % 100000 == 0 else '' for i, x in enumerate(x)])
#plt.xticks([x if int(x[:-1]) % 100 == 0 or x == '1M' else '' for i, x in enumerate(x)])
#plt.xticks([x if i % 100 == 0 else '' for i, x in enumerate(x)])

#plt.show()
plt.savefig(save_path)

#plt.clf()

#plt.plot(train_loss_steps[:100], train_loss_values[:100])
#plt.show()

exit()



##############

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

