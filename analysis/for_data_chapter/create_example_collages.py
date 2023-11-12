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

desired_technique = 'Spinning back kick'
desired_grade = '9 kyu' #   '4 dan'

# ffmpeg -i out%.png -filter_complex tile=1x2 name.png

store_dir = os.path.join(os.getcwd(), 'analysis', 'for_data_chapter', 'collages')

for i, sample in enumerate(data):
    d = sample['joint_positions']
    technique_cls = sample['technique_cls']
    technique = data_info.technique_class_to_name[technique_cls]
    condition = sample['condition']
    grade = sample['grade']

    if condition == 'air' and technique == desired_technique and grade == desired_grade:
        t = desired_technique.replace(' ', '_')
        g = desired_grade.replace(' ', '_')
        file_name = os.path.join(store_dir, f'{t}_{g}_{i}')

        print(f'Index: {i} from {len(data) - 1}')
        print(f'Technique: {technique}')
        print(f'Condition: {condition}')
        print(f'Grade: {grade}')
        from_array(d, mode='collage', file_name=file_name)
        print('------')

# Manually add indices of problematic recordings
# found by this search. Later add these to the outlier modification
# and repeat it on the modified dataset.
found_outliers = []
