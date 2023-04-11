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
# Mirroring along y axis:
# - wrong orientied movements

# pcloud path 
#datapath='/home/anthony/pCloudDrive/storage/data/master_thesis/karate_prep/'

# local path 
datapath='/home/anthony/Documents/ma_data/'

npydatafilepath = os.path.join(datapath, "karate_motion_25_fps.npy")
#npydatafilepath = os.path.join(datapath, "karate_motion_25_fps_modified_outliers.npy")

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

#print(np.max(durations))
#print(upper, lower)

### For checking that the modified dataset contains no outliers anymore
'''
npydatafilepath = os.path.join(datapath, "karate_motion_25_fps_modified_outliers.npy")
data = np.load(npydatafilepath, allow_pickle=True)
positions = [x for x in data["joint_positions"]]
num_frames_in_video = [p.shape[0] for p in positions]
durations = np.array(num_frames_in_video) / framerate
'''
###

duration_outliers = [(i, v, 'duration') for i, v in enumerate(durations) if v >= upper or v <= lower]
print(f'Number of duration outliers: {len(duration_outliers)}')
outliers.extend(duration_outliers)

head_outliers = [(i, durations[i], 'head') for i, d in enumerate(data) if 
    np.min(d['joint_positions'][:, 2]) < 0.5 * np.max(d['joint_positions'][:, 2])]
print(f'Number of head outliers: {len(head_outliers)}')
# Avoiding duplicates
head_outliers = [(i, l, r) for i, l, r in head_outliers if i not in outliers[:][0]]
outliers.extend(head_outliers)

orientation_outliers = [(i, durations[i], 'orientation') for i, d in enumerate(data) if 
    np.mean(d['joint_positions'][:, 1]) > np.mean(d['joint_positions'][:, 7])]
print(f'Number of orientation outliers: {len(orientation_outliers)}')
# Avoiding duplicates
orientation_outliers = [(i, l, r) for i, l, r in orientation_outliers if i not in outliers[:][0]]
outliers.extend(orientation_outliers)

print(f'Total number of outliers: {len(outliers)}')

#exit()

use_report = True

try: 
    report = {}
    if use_report:
        report = json.load(open(os.path.join(datapath, "outlier_report.json"))) 

    new_data = copy.deepcopy(data)
    delete_idxs = []
    for i, (idx, length, reason) in enumerate(outliers):
        pre_mirror = False
        rec = positions[idx]
        original_rec = copy.deepcopy(rec)
        start = 0
        end = int(length * framerate)
        done = False
        if str(idx) in report.keys():
            #actions = list(report[str(idx)]['actions'])
            actions = copy.deepcopy(report[str(idx)]['actions'])
        else: 
            actions = []
        report_step = {} #{'actions': []}
        report_step_done = False
        while not done:
            #print(actions)
            #print('original')
            #mocap_visualization.from_array(original_rec)


            rec = copy.deepcopy(original_rec[start: end])
            if ('mirror' in actions and str(idx) not in report.keys()) or pre_mirror: 
                rec[:, 1::3] *= -1

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
                    action = 'done'
                else:
                    #action = report[str(idx)]['action']
                    action = actions[0]
            else:
                if 'remove' in actions: 
                    action = 'done'
                else:
                    print(f'Length: {rec.shape[0] / framerate} (Outlier border: {upper})')
                    print(f'Outlier reason: {reason}')
                    mocap_visualization.from_array(rec)
                    action = input('Choose an action (remove, mirror, trim) or type done for no further actions: ')

            if action == 'remove':
                #done = True
                #print('removing')
                if str(idx) in report.keys():
                    report_step_done = True
                else:
                    #reason = input('Name a reason: ')
                    #actions.append(action)
                    # When removing there is only one action
                    actions = [action]
                    #report_step['action'] = action #'removed'
                    #report_step['note'] = reason
                    #report[str(idx)] = report_step
                delete_idxs.append(idx)
                #print(f'Finished index {i} of the outliers.')
            elif action == 'mirror':
                #print('morroring')
                rec[:, 1::3] *= -1
                if str(idx) in report.keys():
                    actions.remove(action)
                    pre_mirror = True
                    if len(actions) == 0: 
                        report_step_done = True

                    #mocap_visualization.from_array(rec)
                else:
                    #report_step['action'] = action #'mirror'
                    if action in actions:
                        actions.remove(action)
                    else: 
                        actions.append(action)
            elif action == 'trim':
                #print('trimming')
                if str(idx) in report.keys():
                    new_start_s, new_end_s = report[str(idx)]['borders']
                    new_start = int(new_start_s * framerate)
                    new_end = int(new_end_s * framerate)
                    actions.remove(action)
                    if len(actions) == 0: 
                        report_step_done = True
                    #mocap_visualization.from_array(rec)
                else:
                    new_start_s = float(input('New start (in seconds): '))
                    new_start = int(new_start_s * framerate)
                    new_end_s = float(input('New end (in seconds): '))
                    new_end = int(new_end_s * framerate)
                    new_end = min(new_end, int(length * framerate))
                    print(f'New frame borders: {new_start} and {new_end}')
                    #report_step['action'] = action #'trim' 
                    if action not in actions:
                        actions.append(action)
                    report_step['borders'] = (new_start_s, new_end / framerate)
                start = new_start
                end = new_end
            elif action == 'done': 
                if str(idx) not in report.keys(): 
                    note = input('Pass a note (optional): ')
                    #a = set(actions)
                    a = actions
                    report_step['actions'] = a
                    report_step['note'] = note
                    report[str(idx)] = report_step

                rec_data = copy.deepcopy(data[idx])
                rec_data['joint_positions'] = rec
                rec_data['model_angles'] = rec_data['model_angles'][start: end]
                rec_data['joint_angles'] = new_joint_angles
                rec_data['joint_distances'] = new_joint_distances
                new_data[idx] = rec_data
                done = True
                print(f'Finished index {i} of the outliers (Total: {len(outliers)})')  
                #mocap_visualization.from_array(rec)
            else: 
                if action == 'none' and str(idx) in report.keys():
                    done = True  
                    print(f'Finished index {i} of the outliers.')  
                else: 
                    print('Wrong Input. This recording will be repeated.') 
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
