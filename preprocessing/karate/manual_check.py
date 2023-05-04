#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from visualize.vicon_visualization import from_array
import utils.karate.data_info as data_info
import json


data_dir = os.path.join(os.getcwd(), 'datasets', 'karate')
data_file_path = os.path.join(data_dir, "karate_motion_unmodified.npy")
data = np.load(data_file_path, allow_pickle=True)

use_memory = True
memory_file_dir = os.path.join(os.getcwd(), 'preprocessing', 'karate')
memory_file_path = os.path.join(memory_file_dir, 'check_idx.npy')

if not os.path.exists(memory_file_path):
    np.save(memory_file_path, np.array([0]))

if use_memory:
    start_point = np.load(memory_file_path, allow_pickle=True)[0]
else: 
    start_point = 0

report_dir = os.path.join(os.getcwd(), 'preprocessing', 'karate', 'reports')
report_file_path = os.path.join(report_dir, "outlier_report.json")
if os.path.isfile(report_file_path):
    report = json.load(open(report_file_path))
else:
    print('Warning: Report file not found.')
    report = {}

check_indices = [idx for idx in list(range(start_point, data.shape[0])) if str(idx) not in report.keys()]


for idx in check_indices:
    print(f'Index: {idx}')
    if use_memory:
        np.save(memory_file_path, np.array([idx]))
    
    d = data['joint_positions'][idx]
    technique_cls = data['technique_cls'][idx]
    technique = data_info.technique_class_to_name[technique_cls]
    condition = data['condition'][idx]
    grade = data['grade'][idx]

    print(f'Technique: {technique}')
    print(f'Condition: {condition}')
    print(f'Grade: {grade}')

    from_array(d)
    print('------')

# Manually add indices of problematic recordings
# found by this search. Later add these to the outlier modification
# and repeat it on the modified dataset.
found_outliers = []
