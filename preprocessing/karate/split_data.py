#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import utils.karate.data_info as data_info
import pandas as pd
from io import StringIO
import argparse


def split(arr, cond):
    return arr[cond], arr[~cond]


def get_empty_array():
    j_dist_shape = (len(data_info.reconstruction_skeleton),)
    arr = np.array([], dtype=[
            ('joint_positions', 'O'),
            ('joint_axis_angles', 'O'),
            ('joint_distances', 'f4', j_dist_shape),
            ('attacker_code', 'U10'),
            ('technique_cls', 'i4'),
            ('condition', 'U10'),
            ('age', 'i4'),
            ('gender', 'U10'),
            ('weight', 'i4'),
            ('height', 'i4'),
            ('experience', 'i4'),
            ('grade', 'U10')
        ]
    )
    return arr


def create_split(test_participant_code):
    test_data, train_data = split(data, data['attacker_code'] == test_participant_code)
    validation_data = get_empty_array()

    for p in participants_df.index.tolist():
        for technique in data_info.technique_to_class.keys():
            technique_cls = data_info.technique_to_class[technique]
            data_p_technique, train_data = split(train_data, np.logical_and(train_data['attacker_code'] == p,
                                                 train_data['technique_cls'] == technique_cls))

            # Shuffling so that the validation set does not contain samples
            # from only one condition (air, shield, attacker/defender)
            np.random.shuffle(data_p_technique)

            validation_amount = 2
            # Participant B0370 does not have any samples for the Ushiro-Mawashi-Geri.
            # Therefore, we can not extract any validation samples. To compensate
            # (and still have a balanced validation set), we take one more from participant B0392,
            # and one more from B0405 which have the same grade and the most samples for that technique in that grade.
            if (p == 'B0392' or p == 'B0405') and technique_cls == 4:  # Ushiro-Mawashi-Geri
                validation_amount = 3

            data_p_technique_validation, data_p_technique_train = np.split(data_p_technique, [validation_amount])
            validation_data = np.append(validation_data, data_p_technique_validation)
            train_data = np.append(train_data, data_p_technique_train)

    return train_data, validation_data, test_data


def save_split(train_data, validation_data, test_data, test_participant_code):
    leave_one_out_data_dir = os.path.join(args.target_dir, f'leave_{test_participant_code.lower()}_out')
    if not os.path.exists(leave_one_out_data_dir):
        os.mkdir(leave_one_out_data_dir)

    for name, dataset in [('train', train_data), ('validation', validation_data), ('test', test_data)]:
        file_path = os.path.join(leave_one_out_data_dir, name + '.npy')
        if os.path.isfile(file_path):
            if not args.replace:
                message = 'The target file already exists. If the data should be '
                message += 'replaced by new data, run this script with the --replace argument. Exiting...'
                raise Exception(message)
        np.save(file_path, dataset)
    print(f'Saved split data at {leave_one_out_data_dir}')


def get_args():
    karate_dir = os.path.join(os.getcwd(), 'datasets', 'karate')

    parser = argparse.ArgumentParser(description='Modification of the karate motion data.')
    parser.add_argument('--data_dir', '-d', dest='data_dir', type=str, default=karate_dir,
                        help='The directory that stores the modified karate npy data.')
    parser.add_argument('--target_dir', '-t', dest='target_dir', type=str, default=karate_dir,
                        help='The directory in which the modified data will be stored in.')
    parser.add_argument('--replace', '-r', dest='replace', action='store_true', default=False,
                        help='Set if the existing data in the output directory should be replaced.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    data_dir = args.data_dir
    data_file_path = os.path.join(data_dir, "karate_motion_modified.npy")
    data = np.load(data_file_path, allow_pickle=True)

    cwd = os.getcwd()
    participant_data = open(os.path.join(cwd, 'datasets', 'karate', 'participants.csv'), 'r').readlines()
    # Deleting the unit row.
    del participant_data[1]
    participants_df = pd.read_csv(StringIO(','.join(participant_data)), sep=',', header=0)
    participants_df.set_index('Code', drop=True, inplace=True)

    train_data_1, validation_data_1, b0372_first_dan_test_data = create_split(test_participant_code='B0372')
    train_data_2, validation_data_2, b0401_eighth_kyu_test_data = create_split(test_participant_code='B0401')

    # print(len(train_data_1), len(validation_data_1), len(b0372_first_dan_test_data))
    # print(len(train_data_2), len(validation_data_2), len(b0401_eighth_kyu_test_data))

    save_split(train_data_1, validation_data_1, b0372_first_dan_test_data, 'B0372')
    save_split(train_data_2, validation_data_2, b0401_eighth_kyu_test_data, 'B0401')
