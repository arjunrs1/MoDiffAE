#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from visualize.vicon_visualization import from_array
import utils.karate.data_info as data_info
import json


data_dir = os.path.join(os.getcwd(), 'datasets', 'karate')
data_file_path = os.path.join(data_dir, "karate_motion_modified.npy")
data = np.load(data_file_path, allow_pickle=True)


data = [d for d in data if d['condition'] == 'defender']


for i, sample in enumerate(data):
    d = sample['joint_positions']
    technique_cls = sample['technique_cls']
    technique = data_info.technique_class_to_name[technique_cls]
    condition = sample['condition']
    grade = sample['grade']

    if condition == 'defender':
        print(f'Index: {i} from {len(data) - 1}')
        print(f'Technique: {technique}')
        print(f'Condition: {condition}')
        print(f'Grade: {grade}')
        from_array(d, mode='inspection')
        print('------')

# Manually add indices of problematic recordings
# found by this search. Later add these to the outlier modification
# and repeat it on the modified dataset.
found_outliers = []
