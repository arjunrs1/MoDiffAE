#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import csv
import pandas as pd
from io import StringIO
import numpy as np
import torch
import json

from utils.karate import data_info
from tqdm import tqdm
from visualize import vicon_visualization
from utils.karate import geometry


def create_multi_index_cols(labels, time=True):
    column_combinations = []
    if time:
        column_combinations.append(('Time', ''))
    for la in labels:
        if 'RBAK' in la:
            la = la.replace('RBAK', 'BACK')
        if la != 'Time':
            column_combinations.append((la, 'x'))
            column_combinations.append((la, 'y'))
            column_combinations.append((la, 'z'))
    cols = pd.MultiIndex.from_tuples(column_combinations)
    return cols


def determine_attacker_code(condition, participant_code, xyz_labels):
    # The only case in which the attacker code is not in the file name.
    if condition == 'defender':
        # Extracting attacker code from one of teh labels.
        attacker_code = [la.split(':')[0] for la in xyz_labels if
                         'LFHD' in la and participant_code not in la][0]
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


def interpolate_bad_lines(data, expected_n_cols):
    start = 2
    problematic_row_idxs = []
    invalid = False

    for i in range(start, len(data)):
        # Removing \n
        row = data[i][:-1].split(',')

        if len(row) != expected_n_cols:
            invalid = True
            print('Found column with invalid length')
            problematic_row_idxs.append(i)

            if i == start:
                raise Exception('No previous row available for interpolation')
            elif i == len(data) - 1:
                raise Exception('No next row available for interpolation')

            prev_row = data[i - 1].split(',')
            next_row = data[i + 1].split(',')
            if len(prev_row) != expected_n_cols or len(next_row) != expected_n_cols:
                raise Exception('Multiple consecutive invalid rows')
            prev_row = np.array([float(r) for r in prev_row])
            next_row = np.array([float(r) for r in next_row])

            interpolated_row = np.mean(np.array([prev_row, next_row]), axis=0)
            new_row = [f'{v}' for v in interpolated_row.tolist()]

            new_row = ','.join(new_row) + '\n'
            data[i] = new_row

    return invalid, problematic_row_idxs


def extract_attacker_data(csv_file, n_frames, condition, participant_code, file_name):
    issues = {'n_rows_mismatch': (False, []), 'n_frames_mismatch': False, 'missing_lthi': False}
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
    # Deleting last row if empty.
    if data[-1] == '\n':
        del data[-1]

    invalid, problematic_row_idxs = interpolate_bad_lines(data, expected_n_cols=len(xyz_labels) * 3 + 1)
    if invalid:
        print(file_name)
        issues['n_rows_mismatch'] = (True, problematic_row_idxs)

    df = pd.read_csv(StringIO(','.join(data)), sep=',', header=[0, 1])
    df.columns = cols

    #print(df)

    #print('hi')
    #exit()

    if n_frames != df.shape[0]:
        issues['n_frames_mismatch'] = True
        print(f'Warning: Number of frames {n_frames} in header does not match actual frame number {df.shape[0]}')
        print(file_name)
    df['index'] = np.arange(df.shape[0])
    df = df.set_index('index')

    ds_labels = ['Time']
    if has_codes:
        joint_labels = [attacker_code + ':' + j for j in data_info.joint_to_index.keys()]
    else:
        joint_labels = data_info.joint_to_index.keys()
    ds_labels.extend(joint_labels)

    model_angle_labels = [la for la in xyz_labels if 'Angle' in la and (not has_codes or attacker_code in la)]
    ds_labels.extend(model_angle_labels)

    try:
        df = df[ds_labels]
    except Exception:
        # Known Problem. Some recordings are missing the LTHI marker.
        if 'B0400-S05-E01' in file_name or 'B0400-S05-E02' in file_name:
            issues['missing_lthi'] = True
            print(file_name)
            print('Missing left thigh columns')

            ds_labels = [la for la in ds_labels if 'LTHI' not in la]
            try:
                df = df[ds_labels]
            except Exception:
                raise Exception('Unknown column label error.')
            
            if has_codes:
                lthi_marker_name = attacker_code + ':LTHI'
            else: 
                lthi_marker_name = 'LTHI'

            rthi_marker_name = lthi_marker_name.replace('LTHI', 'RTHI')

            # Placeholder values to create the lthi columns. 
            # Will be replaced later in the code. 
            df.loc[:, (lthi_marker_name, 'x')] = df.loc[:, (rthi_marker_name, 'x')].to_numpy()
            df.loc[:, (lthi_marker_name, 'y')] = df.loc[:, (rthi_marker_name, 'y')].to_numpy()
            df.loc[:, (lthi_marker_name, 'z')] = df.loc[:, (rthi_marker_name, 'z')].to_numpy()

            ds_labels.append(lthi_marker_name)
        else:
            raise Exception('Unknown column label error.')

    if has_codes:
        new_ds_labels = [la.split(':')[1] if la != 'Time' else la for la in ds_labels]
        df.columns = create_multi_index_cols(new_ds_labels)

    return df, attacker_code, issues


def resample_df(df, data_frequency, desired_frequency):
    if desired_frequency > data_frequency:
        raise Exception('Desired frequency (%s) is higher than available (%s).'
                        % (desired_frequency, data_frequency))
    if data_frequency % desired_frequency != 0:
        raise Exception('Desired frequency (%s) is not a factor of the data frequency (%s).'
                        % (desired_frequency, data_frequency))

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


def add_to_sample_list(sample_list, event_dfs, attacker_code, technique_cls, condition, issues, report, file_name):
    age, gender, weight, height, experience, grade = \
        data_info.get_participant_info(attacker_code)

    for e_df in event_dfs:
        e_df = e_df.drop('Time', axis=1, level=0)

        joint_positions = e_df[data_info.joint_to_index.keys()]
        joint_positions = joint_positions.to_numpy().reshape(-1, 39, 3)

        joint_positions = make_skeleton_symmetrical(joint_positions)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        joint_positions_tensor = torch.as_tensor(joint_positions, device=device)
        joint_axis_angles, joint_distances = geometry.calc_axis_angles_and_distances(
            points=joint_positions_tensor
        )

        joint_positions.astype(object)

        joint_axis_angles = joint_axis_angles.cpu().detach().numpy()
        joint_axis_angles.astype(object)

        joint_distances = joint_distances.cpu().detach().numpy()
        joint_distances.astype(object)

        if issues['n_rows_mismatch'][0] or issues['n_frames_mismatch'] or issues['missing_lthi']:
            cur_idx = len(sample_list)
            issues['file_name'] = file_name
            report[str(cur_idx)] = issues

        if np.isnan(joint_positions).any() or \
                np.isnan(joint_axis_angles).any() or \
                np.isnan(joint_distances).any():
            print('Encountered nan value. Exiting...')
            exit()

        sample_list.append((
            joint_positions,
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


'''def evaluate_lthi_approximation(joint_positions, avg_lthi_lkne_dist):
    lthi_idx = data_info.joint_to_index['LTHI']
    lasi_idx = data_info.joint_to_index['LASI']
    lkne_idx = data_info.joint_to_index['LKNE']

    positions = np.array(joint_positions[0])
    count = 1
    for p in joint_positions[1:]:
        positions = np.append(positions, p, axis=0)
        count += 1

    print(f"Used {count} samples for the evaluation of the approximation of the lthi columns")

    lthi_pos_target = positions[:, lthi_idx, :]
    lasi_pos = positions[:, lasi_idx, :]
    lkne_pos = positions[:, lkne_idx, :]

    lasi_pos = np.expand_dims(lasi_pos, axis=0)
    lasi_pos = np.expand_dims(lasi_pos, axis=2)
    lkne_pos = np.expand_dims(lkne_pos, axis=0)
    lkne_pos = np.expand_dims(lkne_pos, axis=2)

    avg_lthi_lkne_dist = np.expand_dims(np.array([avg_lthi_lkne_dist]), axis=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lasi_pos_tensor = torch.as_tensor(lasi_pos, device=device)
    lkne_pos_tensor = torch.as_tensor(lkne_pos, device=device)
    avg_lthi_lkne_dist = torch.as_tensor(avg_lthi_lkne_dist, device=device)

    lasi_lkne_axis_angles, _ = geometry.points_to_axis_angles_and_distances(
        start_points=lkne_pos_tensor,
        end_points=lasi_pos_tensor
    )

    lthi_pos_approx = geometry.axis_angles_and_distances_to_points(
        start_points=lkne_pos_tensor,
        axis_angles=lasi_lkne_axis_angles,
        distances=avg_lthi_lkne_dist
    )

    lthi_pos_target = torch.as_tensor(lthi_pos_target, device=device).squeeze()
    lthi_pos_approx = lthi_pos_approx.squeeze()

    approx_distances = torch.linalg.norm(lthi_pos_target - lthi_pos_approx, dim=-1)
    mean_distance = torch.mean(approx_distances)
    print(f"Expected error of approximation of lthi markers: {mean_distance}mm")
'''

'''def compute_avg_lthi_lkne_dist(joint_positions):

    lthi_idx = data_info.joint_to_index['LTHI']
    lkne_idx = data_info.joint_to_index['LKNE']

    positions = np.array(joint_positions[0])
    for p in joint_positions[1:]:
        positions = np.append(positions, p, axis=0)

    lthi_pos = positions[:, lthi_idx, :]
    lkne_pos = positions[:, lkne_idx, :]

    diff = lthi_pos - lkne_pos
    distances = np.linalg.norm(diff, axis=-1)
    avg_distance = np.average(distances)
    return avg_distance'''


'''def add_lthi_column(df, avg_lthi_lkne_dist):
    lasi_pos = df.loc[:, 'LASI'].to_numpy()
    lkne_pos = df.loc[:, 'LKNE'].to_numpy()

    lasi_pos = np.expand_dims(lasi_pos, axis=0)
    lasi_pos = np.expand_dims(lasi_pos, axis=2)
    lkne_pos = np.expand_dims(lkne_pos, axis=0)
    lkne_pos = np.expand_dims(lkne_pos, axis=2)
    avg_lthi_lkne_dist = np.expand_dims(np.array([avg_lthi_lkne_dist]), axis=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lasi_pos_tensor = torch.as_tensor(lasi_pos, device=device)
    lkne_pos_tensor = torch.as_tensor(lkne_pos, device=device)
    avg_lthi_lkne_dist = torch.as_tensor(avg_lthi_lkne_dist, device=device)

    lasi_lkne_axis_angles, _ = geometry.points_to_axis_angles_and_distances(
        start_points=lkne_pos_tensor,
        end_points=lasi_pos_tensor
    )

    lthi_pos = geometry.axis_angles_and_distances_to_points(
        start_points=lkne_pos_tensor,
        axis_angles=lasi_lkne_axis_angles,
        distances=avg_lthi_lkne_dist
    )
    lthi_pos = lthi_pos.cpu().detach().numpy().squeeze()

    df.loc[:, ('LTHI', 'x')] = lthi_pos[:, 0]
    df.loc[:, ('LTHI', 'y')] = lthi_pos[:, 1]
    df.loc[:, ('LTHI', 'z')] = lthi_pos[:, 2]
    return df'''


def center_initial_position(e_df):
    e_df = e_df.copy()
    # Sternum seems like the most centered joint
    #print(list(e_df.columns))
    #print(e_df.loc[:, 'STRN'].to_numpy())
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


# Interpolation of the 9 markers 
def make_skeleton_symmetrical(joint_positions):
    # shape : (-1, 39, 3)
    for joint_name, (c1_name, c2_name) in data_info.asymmetric_joints_to_neighbours.items():
        c1_idx = data_info.joint_to_index[c1_name]
        c2_idx = data_info.joint_to_index[c2_name]
        joint_idx = data_info.joint_to_index[joint_name]

        c1_data = joint_positions[:, c1_idx, :]
        c2_data = joint_positions[:, c2_idx, :]
        # Calculating midpoints
        joint_data = (c1_data + c2_data) / 2

        joint_positions[:, joint_idx, :] = joint_data

    return joint_positions


def main(desired_frequency, data_dir, target_dir, replace, view_problematic):
    report = {}
    npy_name = 'karate_motion_prepared.npy' #'karate_motion_unmodified.npy'
    file_path = os.path.join(target_dir, npy_name)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if os.path.isfile(file_path):
        if not replace:
            message = 'The target file already exists. If the data should be '
            message += 'replaced by new data, run this script with the --replace argument. Exiting...'
            raise Exception(message)

    # Sorting here is very important. Otherwise, the order might not be the same over different operating systems
    # and storage locations due to the way files are read. This would make indices in the
    # modification incorrect if the data was created on a different machine than the modification is performed.
    file_names = [p for p in sorted(os.listdir(data_dir)) if not p.startswith('.') and not p.endswith('.md')]
    # TODO: remove
    #file_names = file_names[:3]
    
    number_of_files = len(file_names)

    sample_list = []
    #missing_lthi_sample_list = []

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

        df, attacker_code, issues = extract_attacker_data(
            csv_file,
            number_of_frames,
            condition,
            participant_code,
            file_name
        )

        df = resample_df(df, data_frequency, desired_frequency)
        event_dfs = split_events(df, events)

        #if issues['missing_lthi']:
        #    missing_lthi_sample_list.append(
        #        (event_dfs, attacker_code, technique_cls, condition, issues, file_name)
        #    )
        #else:
        event_dfs = center_initial_position_events(event_dfs)
        add_to_sample_list(
            sample_list,
            event_dfs,
            attacker_code,
            technique_cls,
            condition,
            issues,
            report,
            file_name
        )

    

    '''if len(missing_lthi_sample_list) > 0:
        joint_positions_400_s05_e03 = [s[0] for s in sample_list if
                                       s[3] == 'B0400' and s[4] == 4 and s[5] == 'attacker']
        avg_lthi_lkne_dist = compute_avg_lthi_lkne_dist(joint_positions_400_s05_e03)

        evaluate_lthi_approximation(joint_positions_400_s05_e03, avg_lthi_lkne_dist)

        for _, (e_dfs, a_code, tech_cls, cond, issues, file_name) in zip(tqdm(range(len(missing_lthi_sample_list)),
                    desc='Completing samples with missing LTHI marker'), missing_lthi_sample_list):
            completed_dfs = []
            for e_df in e_dfs:
                complete_e_df = add_lthi_column(e_df.copy(), avg_lthi_lkne_dist)
                completed_dfs.append(complete_e_df)
            event_dfs = center_initial_position_events(completed_dfs)
            add_to_sample_list(
                sample_list,
                event_dfs,
                a_code,
                tech_cls,
                cond,
                issues,
                report,
                file_name
            )'''

    j_dist_shape = (len(data_info.reconstruction_skeleton),)
    samples = np.array(sample_list, dtype=[
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

    #vicon_visualization.from_array(samples['joint_positions'][0])
    #vicon_visualization.from_array(samples['joint_positions'][1])
    #exit()

    print(f'Saving processed data at {file_path} ...')
    np.save(file_path, samples)
    print(f'Successfully saved a numpy array with {samples.shape[0]} motion sequences at {desired_frequency} Hz.')

    print(f'Report:')
    for key, value in report.items():
        print(f'{key}: {value}')
    preprocessing_path = os.path.join(os.getcwd(), 'preprocessing', 'karate', 'reports')
    report_path = os.path.join(preprocessing_path, "preparation_report.json")
    with open(report_path, 'w') as outfile:
        json.dump(report, outfile)
    print(f'Saved report at {report_path}.')

    if view_problematic:
        # Inspecting the samples which had problems
        for i in report.keys():
            i = int(i)
            print(f'Showing motion at index {i}...')
            vicon_visualization.from_array(samples['joint_positions'][i])

            axis_angles = np.expand_dims(samples['joint_axis_angles'][i], axis=0)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            axis_angles = torch.as_tensor(axis_angles, device=device)

            start_index = data_info.joint_to_index['T10']
            start = np.expand_dims(samples['joint_positions'][i][:, start_index, :], axis=0)
            start = torch.as_tensor(start, device=device)

            distances = np.expand_dims(samples['joint_distances'][i], axis=0)
            distances = torch.as_tensor(distances, device=device)

            recon_pos = geometry.calc_positions(
                        chain_start_positions=start,
                        start_label='T10',
                        axis_angles=axis_angles,
                        distances=distances
            )
            recon_pos = recon_pos.cpu().detach().numpy().squeeze()

            print('Showing the same motion but reconstructed (should be the same)...')
            vicon_visualization.from_array(recon_pos)


def get_args():
    dataset_dir = os.path.join(os.getcwd(), 'datasets', 'karate')
    #dataset_dir = os.path.join("/home/anthony/pCloudDrive/storage/data/public/motion_diffusion_autoencoder")

    parser = argparse.ArgumentParser(description='Preparation of the karate motion data.')
    parser.add_argument('--data_dir', '-d', dest='data_dir', type=str, default=os.path.join(dataset_dir, 'karate_csv'),
                        help='The directory that stores the karate csv files.')
    parser.add_argument('--target_dir', '-t', dest='target_dir', type=str, default=dataset_dir,
                        help='The directory in which the processed data will be stored in.')
    parser.add_argument('--desired_frequency', '-f', dest='desired_frequency', type=int, default=25,
                        help='The frequency (in Hz) of the output data. \
                            Must be a factor of the original frequency.')
    parser.add_argument('--replace', '-r', dest='replace', action='store_true', default=False,
                        help='Set if the existing data in the output directory should be replaced.')
    parser.add_argument('--view_problematic', '-v', dest='view_problematic', action='store_true', default=False,
                        help='Set to check whether the problematic samples were modified correctly.')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    main(
        args.desired_frequency,
        args.data_dir,
        args.target_dir,
        args.replace,
        args.view_problematic
    )
