#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from io import StringIO


cwd = os.getcwd()
participant_data = open(os.path.join(cwd, 'datasets/karate/participants.csv'), 'r').readlines()
# Deleting the unit row.
del participant_data[1]
participants_df = pd.read_csv(StringIO(','.join(participant_data)), sep=',', header=0)
participants_df.set_index('Code', drop=True, inplace=True)


def get_participant_info(participant_code):
    info = participants_df.loc[participant_code].to_numpy().tolist()
    return info


condition_to_name = {
    'E01': 'air',
    'E02': 'shield',
    'E03': 'attacker',
    'E04': 'defender'
}

technique_to_class = {
    'S01': 0,
    'S02': 1,
    'S03': 2,
    'S04': 3,
    'S05': 4
}

'''technique_class_to_name = {
    0: 'Gyaku-Zuki',   # reverse punch
    1: 'Mae-Geri',   # front kick
    2: 'Mawashi-Geri gedan',   # roundhouse kick at knee to hip height
    3: 'Mawashi-Geri jodan',   # roundhouse kick at shoulder to (top) head height
    4: 'Ushiro-Mawashi-Geri'   # spinning back kick
}'''

technique_class_to_name = {
    0: 'Reverse punch',   # reverse punch
    1: 'Front kick',   # front kick
    2: 'Low roundhouse kick',   # roundhouse kick at knee to hip height
    3: 'High roundhouse kick',   # roundhouse kick at shoulder to (top) head height
    4: 'Spinning back kick'   # spinning back kick
}

asymmetric_joints_to_neighbours = {
    'RUPA': ('RSHO', 'RELB'), 
    'RFRM': ('RELB', 'RWRA'), 
    'RTHI': ('RKNE', 'RPSI'), 
    'RTIB': ('RANK', 'RKNE'), 
    'LUPA': ('LSHO', 'LELB'), 
    'LFRM': ('LELB', 'LWRA'), 
    'LTHI': ('LKNE', 'LPSI'), 
    'LTIB': ('LANK', 'LKNE'), 
    'BACK': ('T10', 'C7')
}

'''joint_to_index_original = {
    'LFHD': 0, 'RFHD': 1, 'LBHD': 2, 'RBHD': 3, 'C7': 4, 'T10': 5, 'CLAV': 6,
    'STRN': 7, 'RBAK': 8, 'LSHO': 9, 'LUPA': 10, 'LELB': 11, 'LFRM': 12, 'LWRA': 13,
    'LWRB': 14, 'LFIN': 15, 'RSHO': 16, 'RUPA': 17, 'RELB': 18, 'RFRM': 19, 'RWRA': 20,
    'RWRB': 21, 'RFIN': 22, 'LASI': 23, 'RASI': 24, 'LPSI': 25, 'RPSI': 26, 'LTHI': 27,
    'LKNE': 28, 'LTIB': 29, 'LANK': 30, 'LHEE': 31, 'LTOE': 32, 'RTHI': 33, 'RKNE': 34,
    'RTIB': 35, 'RANK': 36, 'RHEE': 37, 'RTOE': 38
}'''

joint_to_index = {
    'LFHD': 0, 'RFHD': 1, 'LBHD': 2, 'RBHD': 3, 'C7': 4, 'T10': 5, 'CLAV': 6,
    'STRN': 7, 'BACK': 8, 'LSHO': 9, 'LUPA': 10, 'LELB': 11, 'LFRM': 12, 'LWRA': 13,
    'LWRB': 14, 'LFIN': 15, 'RSHO': 16, 'RUPA': 17, 'RELB': 18, 'RFRM': 19, 'RWRA': 20,
    'RWRB': 21, 'RFIN': 22, 'LASI': 23, 'RASI': 24, 'LPSI': 25, 'RPSI': 26, 'LTHI': 27,
    'LKNE': 28, 'LTIB': 29, 'LANK': 30, 'LHEE': 31, 'LTOE': 32, 'RTHI': 33, 'RKNE': 34,
    'RTIB': 35, 'RANK': 36, 'RHEE': 37, 'RTOE': 38
}

# Assumption: Distance between two connected joints
# does not change (or very little).
# The order can be chosen as desired, as long as the right
# component has already been on the left side previously.
reconstruction_skeleton = [
    # Hip
    ['T10', 'LPSI'],
    ['LPSI', 'RPSI'],
    ['LPSI', 'LASI'],
    ['RPSI', 'RASI'],

    # Left lower body
    ['LPSI', 'LTHI'],
    ['LTHI', 'LKNE'],
    ['LKNE', 'LTIB'],
    ['LTIB', 'LANK'],
    ['LANK', 'LHEE'],
    ['LHEE', 'LTOE'],

    # Right lower body
    ['RPSI', 'RTHI'],
    ['RTHI', 'RKNE'],
    ['RKNE', 'RTIB'],
    ['RTIB', 'RANK'],
    ['RANK', 'RHEE'],
    ['RHEE', 'RTOE'],

    # Center upper body
    ['T10', 'BACK'],
    ['BACK', 'C7'],
    #['T10', 'RBAK'],
    #['RBAK', 'C7'],
    ['C7', 'LBHD'],
    ['C7', 'CLAV'],
    ['CLAV', 'STRN'],

    # Head
    ["LBHD", "RBHD"],
    ["LBHD", "LFHD"],
    ["RBHD", 'RFHD'],

    # Left upper body
    ['C7', 'LSHO'],
    ['LSHO', 'LUPA'],
    ['LUPA', 'LELB'],
    ['LELB', 'LFRM'],
    ['LFRM', 'LWRA'],
    ['LWRA', 'LWRB'],
    ['LWRA', 'LFIN'],

    # Right upper body
    ['C7', 'RSHO'],
    ['RSHO', 'RUPA'],
    ['RUPA', 'RELB'],
    ['RELB', 'RFRM'],
    ['RFRM', 'RWRA'],
    ['RWRA', 'RWRB'],
    ['RWRA', 'RFIN']
]


old_reconstruction_skeleton = [
    # Head
    ['LFHD', 'RFHD'],
    ["LFHD", "LBHD"],
    ["LBHD", "RBHD"],

    # Center upper body
    ['LBHD', 'C7'],
    ['C7', 'RBAK'],
    ['RBAK', 'T10'],
    ['C7', 'CLAV'],
    ['CLAV', 'STRN'],

    # Left upper body
    ['C7', 'LSHO'],
    ['LSHO', 'LUPA'],
    ['LUPA', 'LELB'],
    ['LELB', 'LFRM'],
    ['LFRM', 'LWRA'],
    ['LWRA', 'LWRB'],
    ['LWRA', 'LFIN'],

    # Right upper body
    ['C7', 'RSHO'],
    ['RSHO', 'RUPA'],
    ['RUPA', 'RELB'],
    ['RELB', 'RFRM'],
    ['RFRM', 'RWRA'],
    ['RWRA', 'RWRB'],
    ['RWRA', 'RFIN'],

    # Hip
    ['T10', 'LPSI'],
    ['LPSI', 'RPSI'],
    ['LPSI', 'LASI'],
    ['RPSI', 'RASI'],

    # Left lower body
    ['LASI', 'LTHI'],
    ['LTHI', 'LKNE'],
    ['LKNE', 'LTIB'],
    ['LTIB', 'LANK'],
    ['LANK', 'LHEE'],
    ['LHEE', 'LTOE'],

    # Right lower body
    ['RASI', 'RTHI'],
    ['RTHI', 'RKNE'],
    ['RKNE', 'RTIB'],
    ['RTIB', 'RANK'],
    ['RANK', 'RHEE'],
    ['RHEE', 'RTOE']
]
