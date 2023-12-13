#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from visualize.vicon_visualization import from_array
import utils.karate.data_info as data_info
import json


base_dir = os.path.join(os.getcwd(), 'evaluation', 'test_manipulations')
#data_file_path = os.path.join(data_dir, "karate_motion_modified.npy")
#data = np.load(data_file_path, allow_pickle=True)


for root, dirs, files in os.walk(base_dir):

    trimmed_file_names = [f.split('og_')[-1] if 'og' in f else f.split('manipulated_')[-1] for f in files]
    unique_names = list(set(trimmed_file_names))
    unique_names = [f for f in unique_names if f.endswith('npy')]

    if len(unique_names) > 0:

        # TODO: cretae new combined collage directory if not exists

        for file_name in unique_names:
            og_xyz_path = os.path.join(root, f'og_{file_name}')
            manipulated_xyz_path = os.path.join(root, f'manipulated_{file_name}')
            og_xyz = np.load(og_xyz_path)
            manipulated_xyz = np.load(manipulated_xyz_path)

            og_save_path = og_xyz_path.split(".npy")[0]
            manipulated_save_path = manipulated_xyz_path.split(".npy")[0]

            print(og_save_path)
            from_array(og_xyz, mode='collage', file_name=og_save_path)
            from_array(manipulated_xyz, mode='collage', file_name=manipulated_save_path)

            # TODO: create an new tmp directory.
            #  copy the files there and rename them to be acending.
            #  combine them and store the eusult in th main directoy. then remove tmp.

            exit()

            os.system(f'ffmpeg -i {tmp_folder_name}/out%dts.png -filter_complex tile=10x1 {dir_name}/{base_name_without_extension}.png')

            #from_array(d, mode='inspection')

            print(og_xyz.shape)
            exit()


exit()

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
