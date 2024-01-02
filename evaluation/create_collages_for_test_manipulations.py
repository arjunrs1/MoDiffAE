#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from visualize.vicon_visualization import from_array
import shutil


base_dir = os.path.join(os.getcwd(), 'evaluation', 'test_manipulations')

for root, dirs, files in os.walk(base_dir):

    trimmed_file_names = [f.split('og_')[-1] if 'og' in f else f.split('manipulated_')[-1] for f in files]
    unique_names = list(set(trimmed_file_names))
    unique_names = [f for f in unique_names if f.endswith('npy')]

    if len(unique_names) > 0:

        collage_dir = os.path.join(root, 'combined_collages')
        if not os.path.exists(collage_dir):
            os.mkdir(collage_dir)

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

            tmp_folder_name = os.path.join(collage_dir, f'.tmp_combined_{file_name.split(".npy")[0]}')
            os.mkdir(tmp_folder_name)

            shutil.copy(og_save_path + '.png', os.path.join(tmp_folder_name, 'out1.png'))
            shutil.copy(manipulated_save_path + '.png', os.path.join(tmp_folder_name, 'out2.png'))

            combined_file_name = f'combined_{file_name.split(".npy")[0]}'.replace('_xyz', '')
            os.system(f'ffmpeg -i {tmp_folder_name}/out%d.png -filter_complex tile=1x2 {collage_dir}/{combined_file_name}.png')

            shutil.rmtree(tmp_folder_name)
