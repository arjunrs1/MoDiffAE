#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import csv
import pandas as pd
from io import StringIO
import numpy as np
#import data_info
from utils.karate import data_info
from tqdm import tqdm
#import mocap_visualization
from visualize import vicon_visualization
#import geometry
from utils.karate import geometry


def create_multi_index_cols(labels, time=True):
    column_combinations = []
    if time:
        column_combinations.append(('Time', ''))
    for l in labels:
        if l != 'Time':
            column_combinations.append((l, 'x'))
            column_combinations.append((l, 'y'))
            column_combinations.append((l, 'z'))
    cols = pd.MultiIndex.from_tuples(column_combinations)
    return cols


def determine_attacker_code(condition, participant_code, xyz_labels):
    # The only case in which the attacker code is not in the file name.
    if condition == 'defender':
        # Extracting attacker code from one of teh labels.
        attacker_code = [l.split(':')[0] for l in xyz_labels if
            'LFHD' in l and participant_code not in l][0]
    # The attacker code is in the file name.
    else:
        attacker_code = participant_code
    return attacker_code


def extract_meta_info(csv_file):
    csv_meta_info_reader = csv.reader(csv_file[:6], delimiter=',')
    meta_info = []
    for row in csv_meta_info_reader:
        meta_info.append(row)
    # Deleting an empty row. 
    del meta_info[4]
    return meta_info


def extract_attacker_data(csv_file, number_of_frames, condition, participant_code, file_name):
    print(file_name)
    xyz_labels = list(filter(None, csv_file[7].split(',')))
    # Deleting \n. 
    del xyz_labels[-1]
    # Deleting time label.
    del xyz_labels[0]
    cols = create_multi_index_cols(xyz_labels)

    has_codes = True if condition != 'air' else False
    if has_codes:
        attacker_code = determine_attacker_code(
            condition, participant_code, xyz_labels)
    else:
        attacker_code = participant_code

    data = csv_file[7:]
    # Deleting the unit row.  
    del data[1]
    #df = pd.read_csv(StringIO(','.join(data)), sep=',', header=[0, 1])
    df = pd.read_csv(StringIO(','.join(data)), sep=',', header=[0, 1], on_bad_lines='warn')
    df.columns = cols

    # Creating an index.
    #df['index'] = np.arange(number_of_frames + 1)
    if number_of_frames != df.shape[0] - 1:
        print(f'Warning: Number of frames {number_of_frames} in header does not match actual frame number {df.shape[0] - 1}')
    df['index'] = np.arange(df.shape[0])
    df = df.set_index('index')
    # Deleting the last row because it only has NaN values.
    #df = df.drop(number_of_frames)
    df = df.drop(df.shape[0] - 1)

    ds_labels = ['Time']
    if has_codes:
        joint_labels = [attacker_code + ':' + j for j in data_info.joint_to_index.keys()]
    else:
        joint_labels = data_info.joint_to_index.keys()
    ds_labels.extend(joint_labels)

    model_angle_labels = [l for l in xyz_labels if 'Angle' in l and (not has_codes or attacker_code in l)]
    ds_labels.extend(model_angle_labels)

    missing_lthi = False
    try:
        df = df[ds_labels]
    except Exception:
        # Known Problem. Some recordings are missing the LTHI marker.
        if 'B0400-S05-E01' in file_name or 'B0400-S05-E02' in file_name:
            missing_lthi = True
            ds_labels = [l for l in ds_labels if 'LTHI' not in l]
            df = df[ds_labels]
        else:
            raise Exception('Unknown column label error.')

    if has_codes:
        new_ds_labels = [l.split(':')[1] if l != 'Time' else l for l in ds_labels]
        df.columns = create_multi_index_cols(new_ds_labels)

    return df, attacker_code, missing_lthi


def resample_df(df, data_frequency, desired_frequency):
    if desired_frequency > data_frequency:
        raise Exception('Desired frequency (%s) is higher than available (%s).'
            %(desired_frequency, data_frequency))
    if data_frequency % desired_frequency != 0:
        raise Exception('Desired frequency (%s) is not a factor of the data frequency (%s).'
            %(desired_frequency, data_frequency))

    step_size = int(data_frequency / desired_frequency)
    indices = list(range(0, len(df.index), step_size))

    resampled_df = df.loc[indices, :]
    resampled_df.reset_index(drop=True, inplace=True)

    return resampled_df


def split_events(df, events):
    events_dfs = []

    for i, event_start in enumerate(events[:-1]):
        event_end = events[i+1]
        mask = (df['Time'] >= event_start) & (df['Time'] <= event_end)
        event_df = df.loc[mask]
        event_df.reset_index(drop=True, inplace=True)
        events_dfs.append(event_df)

    return events_dfs


def add_to_sample_list(sample_list, event_dfs, attacker_code, technique_cls, condition):
    age, gender, weight, height, experience, grade = \
        data_info.get_participant_info(attacker_code)

    model_angle_labels = list(dict.fromkeys([c[0] for c in event_dfs[0].columns if 'Angle' in c[0]]))

    for e_df in event_dfs:
        e_df = e_df.drop('Time', axis=1, level=0)

        joint_positions = e_df[data_info.joint_to_index.keys()]
        joint_axis_angles, joint_distances = geometry.calc_axis_angles_and_distances(
            joint_positions_df=joint_positions
        )
        joint_positions = joint_positions.to_numpy()
        joint_positions.astype(object)

        model_angles = e_df[model_angle_labels]
        model_angles = model_angles.to_numpy()
        model_angles.astype(object)

        sample_list.append((
            joint_positions,
            model_angles,
            joint_axis_angles,
            joint_distances,
            attacker_code,
            technique_cls,
            condition,
            age,
            gender,
            weight,
            height,
            experience,
            grade
        ))

def compute_avg_LTHI_LKNEE_dist(joint_positions):
    LTHI_idx = data_info.joint_to_index['LTHI'] * 3
    LKNE_idx = data_info.joint_to_index['LKNE'] * 3

    positions = np.array(joint_positions[0])
    for p in joint_positions[1:]:
        positions = np.append(positions, p, axis=0)

    LTHI_pos = positions[:, LTHI_idx:LTHI_idx+3]
    LKNE_pos = positions[:, LKNE_idx:LKNE_idx+3]

    diff = LTHI_pos - LKNE_pos
    distances = np.linalg.norm(diff, axis=1)
    avg_distance = np.average(distances)
    return avg_distance

def add_LTHI_column(df, avg_LTHI_LKNE_dist):
    LASI_pos = df.loc[:, 'LASI'].to_numpy()
    LKNE_pos = df.loc[:, 'LKNE'].to_numpy()

    LTHI_pos = np.zeros(shape=LKNE_pos.shape)
    for t in range(LKNE_pos.shape[0]):
        LASI_LKNE_axis_angle, _ = geometry.points_to_axis_angle_and_distance(
            start_point=LKNE_pos[t],
            end_point=LASI_pos[t]
        )
        LTHI_pos[t] = geometry.axis_angle_and_distance_to_point(
            start_point=LKNE_pos[t],
            axis_angle=LASI_LKNE_axis_angle,
            distance=avg_LTHI_LKNE_dist
        )

    df.loc[:, ('LTHI', 'x')] = LTHI_pos[:, 0]
    df.loc[:, ('LTHI', 'y')] = LTHI_pos[:, 1]
    df.loc[:, ('LTHI', 'z')] = LTHI_pos[:, 2]
    return df

def center_initial_position(e_df):
    e_df = e_df.copy()
    # Sternum seems like the most centered joint
    offset = e_df.loc[0, 'STRN'].to_numpy()
    for j in data_info.joint_to_index.keys():
        e_df.loc[:, (j, 'x')] = e_df.loc[:, (j, 'x')] - offset[0]
        e_df.loc[:, (j, 'y')] = e_df.loc[:, (j, 'y')] - offset[1]
    return e_df

def center_initial_position_events(event_dfs):
    centered_event_dfs = []
    for e_df in event_dfs:
        centered_df = center_initial_position(e_df)
        centered_event_dfs.append(centered_df)
    return centered_event_dfs

def main(desired_frequency, data_dir, output_dir, replace):
    npy_name = f'karate_motion_{desired_frequency}_fps_axis_angles_t10_test.npy'
    file_path = os.path.join(output_dir, npy_name)

    print(desired_frequency)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if os.path.isfile(file_path):
        if not replace:
            message = 'The target file already exists. If the data should be '
            message += 'replaced by new data, run this script with the --replace argument. Exiting...'
            raise Exception(message)

    file_names = [dir for dir in os.listdir(data_dir) if not dir.startswith('.')]
    # TODO: remove
    file_names = file_names[:10]

    number_of_files = len(file_names)

    sample_list = []
    missing_LTHI_sample_list = []

    for _, file_name in zip(tqdm(range(number_of_files), desc='Processing motion data'), file_names):
        full_path = os.path.join(data_dir, file_name)
        csv_file = open(full_path, 'r').readlines()

        meta_info = extract_meta_info(csv_file)

        number_of_frames = int(meta_info[0][1])
        data_frequency = int(meta_info[2][1])

        events = [float(e) for e in meta_info[4][1:]]

        file_info = file_name.split('.')[0].split('-')
        condition = data_info.condition_to_name[file_info[5]]
        participant_code = file_info[3]
        technique_cls = data_info.technique_to_class[file_info[4]]

        df, attacker_code, missing_LTHI = extract_attacker_data(
            csv_file,
            number_of_frames,
            condition,
            participant_code,
            file_name
        )

        df = resample_df(df, data_frequency, desired_frequency)
        event_dfs = split_events(df, events)

        if missing_LTHI:
            missing_LTHI_sample_list.append(
                (event_dfs, attacker_code, technique_cls, condition)
            )
        else:
            event_dfs = center_initial_position_events(event_dfs)
            add_to_sample_list(
                sample_list,
                event_dfs,
                attacker_code,
                technique_cls,
                condition
            )

    if len(missing_LTHI_sample_list) > 0:
        joint_positions_400_S05_E03 = [s[0] for s in sample_list if \
            s[4] == 'B0400' and s[5] == 4 and s[6] == 'attacker']
        avg_LTHI_LKNE_dist = compute_avg_LTHI_LKNEE_dist(joint_positions_400_S05_E03)
        for _, (e_dfs, a_code, tech_cls, cond) in zip(tqdm(range(len(missing_LTHI_sample_list)),
                desc='Completing samples with missing LTHI marker'), missing_LTHI_sample_list):
            completed_dfs = []
            for e_df in e_dfs:
                complete_e_df = add_LTHI_column(e_df.copy(), avg_LTHI_LKNE_dist)
                completed_dfs.append(complete_e_df)
            event_dfs = center_initial_position_events(completed_dfs)
            #add_to_sample_list(
            #    sample_list,
            #    completed_dfs,
            #    a_code,
            #    tech_cls,
            #    cond
            #)
            add_to_sample_list(
                sample_list,
                event_dfs,
                a_code,
                tech_cls,
                cond
            )

    j_dist_shape = (len(data_info.reconstruction_skeleton),)
    samples = np.array(sample_list , dtype=[
            ('joint_positions', 'O'),
            ('model_angles', 'O'),
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


    #print(f'Saving processed data at {file_path} ...')
    #np.save(file_path, samples)
    #print(f'Successfully saved a numpy array with {samples.shape[0]} motion sequences at {desired_frequency} Hz.')

    #'''
    for i in range(samples['joint_positions'].shape[0]):
        print(i)
        print('Showing exemplary motion...')
        vicon_visualization.from_array(samples['joint_positions'][i])

        axis_angles = samples['joint_axis_angles'][i]
        start_index = data_info.joint_to_index['T10']
        start = samples['joint_positions'][i][:, start_index*3:start_index*3+3]
        distances = samples['joint_distances'][i]
        recon_pos = geometry.calc_positions(
                    chain_start_positions=start,
                    start_label='T10', #"LFHD",
                    axis_angles=axis_angles,
                    distances=distances
        )
        print('Showing the same motion but reconstructed (should be the same)...')
        vicon_visualization.from_array(recon_pos)
    #'''

def get_args():
    parser = argparse.ArgumentParser(description='Preparation of the karate motion data.')
    parser.add_argument('--data_dir', '-d', dest='data_dir', type=str,
        default='/home/anthony/pCloudDrive/storage/data/master_thesis/karate_csv/',
        help='The directory that stores the karate csv files.')
    parser.add_argument('--target_dir', '-t', dest='target_dir', type=str,
        default='/home/anthony/pCloudDrive/storage/data/master_thesis/karate_prep/',
        help='The directory in which the processed data will be stored in.')
    parser.add_argument('--desired_frequency', '-f', dest='desired_frequency', type=int, default=25,
                        help='The frequency (in Hz) of the output data. \
                            Musst be a factor of the original frequency.')
    parser.add_argument('--replace', '-r', dest='replace', action='store_true', default=False,
        help='Set if the existing data in the output directory should be replaced.')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    main(
        args.desired_frequency,
        args.data_dir,
        args.target_dir,
        args.replace
    )
