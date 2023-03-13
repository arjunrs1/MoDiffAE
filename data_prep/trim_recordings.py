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


datapath='/home/anthony/pCloudDrive/storage/data/master_thesis/karate_prep/'
npydatafilepath = os.path.join(datapath, "karate_motion_25_fps.npy")
data = np.load(npydatafilepath, allow_pickle=True)

positions = [x for x in data["joint_positions"]]
num_frames_in_video = [p.shape[0] for p in positions]

outliers = []
framerate = 25

durations = np.array(num_frames_in_video) / framerate
mean = np.mean(durations) 
std = np.std(durations)
upper = mean + 3 * std
lower = mean - 3 * std
duration_outliers = [(i, v) for i, v in enumerate(durations) if v >= upper or v <= lower]
print(f'Number of duration outliers: {len(duration_outliers)}')
outliers.extend(duration_outliers)

head_outliers = [(i, durations[i]) for i, d in enumerate(data) if 
    np.min(d['joint_positions'][:, 2]) < 0.5 * np.max(d['joint_positions'][:, 2])]
print(f'Number of head outliers: {len(head_outliers)}')
outliers.extend(head_outliers)

print(f'Total number of outliers: {len(outliers)}')

###

base_file_name = '/home/anthony/Pictures/'

'''advanced_mae_geri_air = [(i, d) for i, d in enumerate(data) if 
    d['technique_cls'] == 1 and d['condition'] == 'air' and d['grade'] == '4 dan']

print(len(advanced_mae_geri_air))
mocap_visualization.from_array(advanced_mae_geri_air[-1][1]['joint_positions'], file_name=base_file_name + 'adv')

#


beginner_mae_geri_air = [(i, d) for i, d in enumerate(data) if 
    d['technique_cls'] == 1 and d['condition'] == 'air' and d['grade'] == '9 kyu']

print(len(beginner_mae_geri_air))
mocap_visualization.from_array(beginner_mae_geri_air[0][1]['joint_positions'], file_name=base_file_name + 'beg.ogv')
'''
#

for i in range(len(head_outliers)): 
    mocap_visualization.from_array(data[head_outliers[i][0]]['joint_positions'], file_name=base_file_name + f'fail{i}')

exit()







###

use_report = True

try: 
    report = {}
    if use_report:
        report = json.load(open(os.path.join(datapath, "outlier_report.json"))) 

    new_data = copy.deepcopy(data)
    delete_idxs = []
    for i, (idx, length) in enumerate(outliers):
        rec = positions[idx]
        original_rec = rec
        start = 0
        end = int(length * framerate)
        done = False
        report_step = {'action': 'none'}
        report_step_done = False
        while not done:
            rec = original_rec[start: end]

            # Recentering and calculating new joint angles and distances
            positions_df = pd.DataFrame(rec, 
                columns=data_prep.create_multi_index_cols(data_info.joint_to_index.keys(), time=False))
            centered_positions_df = data_prep.center_initial_position(positions_df)
            new_joint_angles, new_joint_distances = geometry.calc_joint_angles_and_distances(
                joint_positions_df=centered_positions_df
            )

            rec = centered_positions_df.to_numpy()
            rec.astype(object)
            #test_recon = geometry.calc_positions(
            #    rec[:, 0:3], 'LFHD', new_joint_angles, new_joint_distances)
            #mocap_visualization.from_array(test_recon)

            if str(idx) in report.keys():
                if report_step_done:
                    was_ok = ''
                else:
                    action = report[str(idx)]['action']
                    if action == 'none': 
                        was_ok = ''
                    else: 
                        was_ok = 'no'
            else:
                print(f'Length: {rec.shape[0] / framerate}')
                mocap_visualization.from_array(rec)
                was_ok = input('Everything ok? Type no or a note to save it and continue: ')
            if was_ok == 'no':
                if str(idx) in report.keys():
                    action = report[str(idx)]['action']
                    if action == 'removed': 
                        remove = 'yes'
                    else: 
                        remove = 'no'
                else:
                    remove = input('Should the recording be removed? Type yes or no: ')

                if remove == 'yes': 
                    done = True
                    if str(idx) not in report.keys():
                        reason = input('Name a reason: ')
                        report_step['action'] = 'removed'
                        report_step['note'] = reason
                        report[str(idx)] = report_step
                    delete_idxs.append(idx)
                    print(f'Finished index {i} of the outliers.')
                elif remove == 'no':
                    if str(idx) in report.keys():
                        report_step_done = True
                        new_start_s, new_end_s = report[str(idx)]['borders']
                        new_start = int(new_start_s * framerate)
                        new_end = int(new_end_s * framerate)
                    else: 
                        new_start_s = float(input('New start (in seconds): '))
                        new_start = int(new_start_s * framerate)
                        new_end_s = float(input('New end (in seconds): '))
                        new_end = int(new_end_s * framerate)
                        new_end = min(new_end, int(length * framerate))
                        print(f'New frame borders: {new_start} and {new_end}')
                        report_step['action'] = 'cut' 
                        report_step['borders'] = (new_start_s, new_end / framerate)
                    start = new_start
                    end = new_end
                else: 
                    print('Wrong Input. This recording will be repeated.') 
            else: 
                done = True
                if str(idx) not in report.keys(): 
                    try:
                        note = report_step['note']
                        report_step['note'] = note + was_ok
                    except KeyError:
                        report_step['note'] = was_ok

                    report[str(idx)] = report_step 

                rec_data = copy.deepcopy(data[idx])
                rec_data['joint_positions'] = rec
                rec_data['model_angles'] = rec_data['model_angles'][start: end]
                rec_data['joint_angles'] = new_joint_angles
                rec_data['joint_distances'] = new_joint_distances
                new_data[idx] = rec_data
                print(f'Finished index {i} of the outliers.')
except KeyboardInterrupt:
    print(f'\nAugmentation was interrupted at index {i} of the outliers.')
finally: 
    new_data = [x for (i, x) in enumerate(new_data) if i not in delete_idxs]
    new_npydatafilepath = os.path.join(datapath, "karate_motion_25_fps_modified_outliers.npy")
    np.save(new_npydatafilepath, new_data)
    print(f'Saved new data at {new_npydatafilepath}.')
    print(f'Report:')
    for key, value in report.items():
        print(f'{key}: {value}')

    report_path = os.path.join(datapath, "outlier_report.json")
    print(f'Saved report at {report_path}.')
    with open(report_path, 'w') as outfile: 
        json.dump(report, outfile)
