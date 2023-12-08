#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from visualize.vicon_visualization import from_array
import utils.karate.data_info as data_info
import json


data_dir = os.path.join(os.getcwd(), 'datasets', 'karate', 'leave_b0372_out')
data_file_path = os.path.join(data_dir, "generated_data_33_percent_interrupted.npy")
data = np.load(data_file_path, allow_pickle=True)

print(data.shape)

for i, sample in enumerate(data):
    d = sample['joint_positions']
    technique_cls = sample['technique_cls']
    technique = data_info.technique_class_to_name[technique_cls]
    grade = sample['grade']

    print(f'Index: {i} from {len(data) - 1}')
    print(f'Technique: {technique}')
    print(f'Grade: {grade}')
    from_array(d, mode='inspection')
    print('------')

