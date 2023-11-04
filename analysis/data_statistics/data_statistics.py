#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from visualize.vicon_visualization import from_array
import utils.karate.data_info as data_info
import json
import csv


def calc_technique_statistics():

    codes = list(set(list(data['attacker_code'])))
    codes.sort()

    codes_with_data = [(p, [d for d in data if d['attacker_code'] == p]) for p in codes]

    technique_classes = list(data_info.technique_class_to_name.keys())
    print(technique_classes)

    print(len(data))

    print(codes)

    #print(len([d for d in data if d['condition'] == 'attacker']))
    #exit()

    with open(os.path.join(os.getcwd(), 'analysis', 'data_statistics', 
                           'technique_statistics.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Code", *data_info.technique_class_to_name.values(), 'Total per condition', 'Total'])


        for p, d in codes_with_data:
            row = [p]
            sum_p_air = 0
            sum_p_shield = 0
            sum_p_opponent = 0
            for cls in technique_classes:
                technique_d_air = [t_d for t_d in d if t_d['technique_cls'] == cls 
                                   and t_d['condition'] == 'air']
                technique_d_shield = [t_d for t_d in d if t_d['technique_cls'] == cls 
                                      and t_d['condition'] == 'shield']
                technique_d_opponent = [t_d for t_d in d if t_d['technique_cls'] == cls 
                                        and (t_d['condition'] == 'attacker' or t_d['condition'] == 'defender')] 
                sum_p_air += len(technique_d_air)
                sum_p_shield += len(technique_d_shield)
                sum_p_opponent += len(technique_d_opponent)
                row.append(f'{len(technique_d_air)}(A), {len(technique_d_shield)}(S), {len(technique_d_opponent)}(O)')
            row.append(f'{sum_p_air}(A), {sum_p_shield}(S), {sum_p_opponent}(O)')
            row.append(f'{sum_p_air + sum_p_shield + sum_p_opponent}')
            writer.writerow(row)
            
    exit()
    for d in data:
        print(d['technique_cls'])
    


if __name__ == '__main__':
    # For data statistcs
    data_dir = os.path.join(os.getcwd(), 'datasets', 'karate')
    data_file_path = os.path.join(data_dir, "karate_motion_modified.npy")
    try:
        data = np.load(data_file_path, allow_pickle=True)
    except Exception:
        print('Data file not found.')
        exit()

    # For outlier modification statistcs
    report_dir = os.path.join(os.getcwd(), 'preprocessing', 'karate', 'reports')
    report_file_path = os.path.join(report_dir, "outlier_report.json")
    try:
        report = json.load(open(report_file_path))
    except Exception:
        print('Report file not found.')
        exit()

    # For removed duplicate statistcs
    removed_duplicates_file_path = os.path.join(report_dir, "removed_duplicates.json")
    try:
        removed_duplicates = json.load(open(removed_duplicates_file_path))
    except Exception:
        print('No list of removed duplicates found.')
        exit()


    calc_technique_statistics()

    exit()
    #check_indices = [idx for idx in list(range(start_point, data.shape[0]))
    #                 if str(idx) not in report.keys() and idx not in removed_duplicates]


    print(f'{len(check_indices)} samples remaining')

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
