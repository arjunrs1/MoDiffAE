#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import mocap_visualization
import copy
import data_prep
import geometry
import pandas as pd
import data_info
import json

# Cutting: 
# - Movement at beginning or end that is unrelated to the technique
# - Too long paused at the end or beginning (especially for outliers)
# Removing: 
# - faulires, e.g. falling
# Reason why cutting can not be optimized well: Sometimes participants walk around 
# or bend down etc. So it is not always a pause / no movement.  

# pcloud path 
#datapath='/home/anthony/pCloudDrive/storage/data/master_thesis/karate_prep/'

# local path 
datapath='/home/anthony/Documents/ma_data/'

npydatafilepath = os.path.join(datapath, "karate_motion_25_fps_modified_outliers.npy")
data = np.load(npydatafilepath, allow_pickle=True)

# Notes: 
# - Initial idea: Result: all defenrders are wrong oriented
# assuming they followed a strict protocol
# Result: condition == defender does not always mean 
# that the orientation is wrong. 
#  Not a good criterium. (1086)
# - instead try: avg y values are on te wrong side (above 0)
# Result: better but still some falts. Espesialy with kicks, often 
# the mean is in the positive y area due to leaning the torso in the oposite direction 
# of the kick. (799 detections)
# - next idea: max is further away from center than min. Inspired by typical 
# forward motion in karate movements.
# Still not perfect but better (165 detections, the kick lean problem not there)
# - simply check if the front head is on the correct side of the back head 
# at the beginning (92 outliers). 
# After manuyll going through thm to confirm them, 
# the only few falts that were wrongly detected are ones where
# paticipants started in the wrong orientation and then turned 
# around. It should also be mentioned that this could also happen the other way around (people starting right and then turn
# the wrong way). This motivated the final idea:
# Next idea: avg. 85 outliers. No false detections. (and one false recording where nothing happens..)

# Nevertheless, check all recording after modification.


use_memory = True
memory_file_path = os.path.join(datapath, 'check_idx.npy')

if not os.path.exists(memory_file_path):
    np.save(memory_file_path, np.array([0]))

if use_memory:
    start_point = np.load(memory_file_path, allow_pickle=True)[0]
else: 
    start_point = 0


# TODO: add indices of problematic recordings
# found by this manual search.
# Also recordings with wrong technique and 
# then ask felix if these should be cut. 
# Argument: maybe cut only at the point where they 
# tought the start position would be (even if wrong foot placemenet)
# Only remove things unrelated to technique (?)
found_outliers = []



#for idx, (i, d) in enumerate(orientation_outliers):
for idx in range(start_point, data.shape[0]):


    print(f'Index: {idx}')
    np.save(memory_file_path, np.array([idx]))
    
    d = data['joint_positions'][idx]
    technique_cls = data['technique_cls'][idx]
    technique = data_info.technique_class_to_name[technique_cls]
    condition = data['condition'][idx]
    grade = data['grade'][idx]

    # 'attacker'
    # 'air' 
    # 'shield'
    # 'defender'
    #if technique == 'Ushiro-Mawashi-Geri' and condition == 'defender':
    print(technique)
    print(condition)
    print(grade)
    #mocap_visualization.from_array(d)

    #d_new = d
    #d_new[:, 1::3] *= -1

    mocap_visualization.from_array(d)
    print('------')



    