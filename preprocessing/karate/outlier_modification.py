#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from visualize.vicon_visualization import from_array
import copy
import preprocessing.karate.data_prep as data_prep
import utils.karate.geometry as geometry
import pandas as pd
import utils.karate.data_info as data_info
import json
import torch
import argparse


def add_to_outliers(old_outliers, new_outliers):
    for i in range(len(new_outliers)):
        index, l, r = new_outliers[i]
        for k in range(len(old_outliers)):
            other_index, _, other_r = old_outliers[k]
            if other_index == index:
                old_outliers[k] = (index, l,  other_r + f', {r}')
    new_outliers = [(i, l, r) for i, l, r in new_outliers if i not in [item[0] for item in old_outliers]]
    old_outliers.extend(new_outliers)


def add_duration_outliers():
    for cl in data_info.technique_class_to_name.keys():
        cls_durations = [(i, p.shape[0] / frequency) for i, (p, cls) in
                         enumerate(zip(positions, [c for c in data['technique_cls']])) if cls == cl]
        dur = [d for i, d in cls_durations]
        mean = np.mean(dur)
        std = np.std(dur)
        cls_name = data_info.technique_class_to_name[cl]
        if cls_name == 'Spinning back kick': #'Ushiro-Mawashi-Geri': 
            # Longest and most difficult technique gets stricter criterion
            # (Since one goal is to reduce the maximum duration).
            z = 1.5
        else:
            z = 3
        upper = mean + z * std
        lower = mean - z * std
        duration_outliers = [(i, v, f'duration ({cls_name}, borders: ({lower:.2f}, {upper:.2f}))')
                             for i, v in cls_durations if v >= upper or v <= lower]
        print(f'Number of duration outliers for the {data_info.technique_class_to_name[cl]}: '
              f'{len(duration_outliers)} (borders: ({lower:.2f}, {upper:.2f}))')
        #outliers.extend(duration_outliers)
        add_to_outliers(outliers, duration_outliers)


def add_head_outliers():
    lfhd_idx = data_info.joint_to_index['LFHD']
    head_outliers = [(i, durations[i], 'head') for i, d in enumerate(data) if
                     np.min(d['joint_positions'][:, lfhd_idx, 2]) < 0.5 * np.max(d['joint_positions'][:, lfhd_idx, 2])]
    print(f'Number of head outliers: {len(head_outliers)}')
    add_to_outliers(outliers, head_outliers)


def add_orientation_outliers():
    lfhd_idx = data_info.joint_to_index['LFHD']
    lbhd_idx = data_info.joint_to_index['LBHD']
    orientation_outliers = [(i, durations[i], 'orientation') for i, d in enumerate(data) if
                            np.mean(d['joint_positions'][:, lfhd_idx, 1]) >
                            np.mean(d['joint_positions'][:, lbhd_idx, 1])]
    print(f'Number of orientation outliers: {len(orientation_outliers)}')
    add_to_outliers(outliers, orientation_outliers)


def add_dominant_side_outliers():
    left_hand_labels = ['LFIN']
    right_hand_labels = ['RFIN']
    left_hand_indices = [data_info.joint_to_index[la] for la in left_hand_labels]
    right_hand_indices = [data_info.joint_to_index[ra] for ra in right_hand_labels]

    left_foot_labels = ['LTOE', 'LANK', 'LHEE']
    right_foot_labels = ['RTOE', 'RANK', 'RHEE']
    left_foot_indices = [data_info.joint_to_index[lf] for lf in left_foot_labels]
    right_foot_indices = [data_info.joint_to_index[rf] for rf in right_foot_labels]

    velocities = [d['joint_positions'][1:, :, :] - d['joint_positions'][:-1, :, :] for d in data]

    #left_finger_distances_from_center = [
    #    np.squeeze(np.linalg.norm(d['joint_positions'][:, left_hand_indices, 1] - 
    #                   np.zeros_like(d['joint_positions'][:, left_hand_indices, 1]), axis=-1)) for d in data]

    #right_finger_distances_from_center = [
    #    np.squeeze(np.linalg.norm(d['joint_positions'][:, right_hand_indices, 1] - 
    #                   np.zeros_like(d['joint_positions'][:, right_hand_indices, 1]), axis=-1)) for d in data]

    #dominant_side_hand_outliers = [(i, durations[i], 'left hand dominant') for i, (d, vel) in enumerate(zip(data, velocities)) if
    #                        d['technique_cls'] == 0 and np.argmax(left_finger_distances_from_center[i]) > np.argmax(right_finger_distances_from_center[i])]

    # In the reverse punch, the dominant hand moves later than the non-dominant
    dominant_side_hand_outliers = [(i, durations[i], 'left hand dominant') for i, (d, vel) in enumerate(zip(data, velocities)) if
                            d['technique_cls'] == 0 and np.argmax(vel[:, left_hand_indices, :]) > np.argmax(vel[:, right_hand_indices, :])]


    dominant_side_foot_outliers = [(i, durations[i], 'left foot dominant') for i, (d, vel) in enumerate(zip(data, velocities)) if
                            d['technique_cls'] != 0 and np.max(vel[:, left_foot_indices, :]) > np.max(vel[:, right_foot_indices, :])]

    print(f'Number of dominant side outliers: {len(dominant_side_hand_outliers) + len(dominant_side_foot_outliers)}')
    add_to_outliers(outliers, dominant_side_hand_outliers)
    add_to_outliers(outliers, dominant_side_foot_outliers)


def add_no_movement_outliers():
    mean_variances = [np.mean(np.var(d['joint_positions'], axis=0)) for d in data]
    mean_variances_mean = np.mean(mean_variances)
    mean_variances_std = np.std(mean_variances)
    mean_variances_lower = mean_variances_mean - 1.8 * mean_variances_std
    no_movement_outliers = [(i, durations[i], f'no movement (lower variance border: {mean_variances_lower})')
                            for i, d in enumerate(data) if
                            np.mean(np.var(d['joint_positions'], axis=0)) < mean_variances_lower]
    print(f'Number of no movement outliers: {len(no_movement_outliers)} '
          f'(lower variance border: {mean_variances_lower:.2f})')
    add_to_outliers(outliers, no_movement_outliers)


def add_t_pose_outliers():
    arm_labels = ['LSHO', 'LUPA', 'LELB', 'LFRM', 'LWRA', 'LWRB', 'LFIN']
    arm_labels.extend(['RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN'])
    arm_indices = [data_info.joint_to_index[la] for la in arm_labels]
    # Determining arm y variance border
    arms_y_values = [d['joint_positions'][:, arm_indices, 1] for d in data]
    y_variances = [np.var(d, axis=-1) for d in arms_y_values]
    y_variances_mean = np.mean([np.mean(y_v) for y_v in y_variances])
    y_variances_std = np.std([np.std(y_v) for y_v in y_variances])
    y_variances_lower = y_variances_mean - 1.8 * y_variances_std # 1.8
    # Determining arm z variance border
    arms_z_values = [d['joint_positions'][:, arm_indices, 2] for d in data]
    z_variances = [np.var(d, axis=-1) for d in arms_z_values]
    z_variances_mean = np.mean([np.mean(z_v) for z_v in z_variances])
    z_variances_std = np.std([np.std(z_v) for z_v in z_variances])
    z_variances_lower = z_variances_mean - 1.3 * z_variances_std # 1.4
    # Only an outlier if all arm markers are at a similar height (z) and width (y)
    # and this happens in a time window of a second.
    t_pose_outliers = [(i, durations[i], 't-pose')
                       for i, (y_v, z_v) in enumerate(zip(y_variances, z_variances)) if
                       np.min(y_v) < y_variances_lower and np.min(z_v) < z_variances_lower
                       and np.isclose(np.argmin(y_v), np.argmin(z_v), atol=frequency)]
    print(f'Number of t-pose outliers: {len(t_pose_outliers)} '
          f'(lower x and y arm variance borders: ({y_variances_lower:.2f}, {z_variances_lower:.2f}))')
    add_to_outliers(outliers, t_pose_outliers)


def add_manual_outliers():
    # This list contains all the indices found during the manual search
    manual_indices = []
    manual_outliers = [(i, durations[i], 'manually found') for i in manual_indices]
    print(f'Number of additional outliers found with manual search: {len(manual_outliers)}')
    add_to_outliers(outliers, manual_outliers)


def find_duplicate_sample_indices():
    indices = []
    joint_positions = [d['joint_positions'] for d in data]
    for i in range(len(data)):
        j_p = joint_positions[i]
        equals = [e for e in joint_positions if j_p.shape == e.shape and np.all((j_p == e))]
        if len(equals) == 2 and data[i]['condition'] == 'defender':
            indices.append(i)
        elif len(equals) == 2 and data[i]['condition'] != 'attacker':
            raise Exception('Unexpected duplicate.')
        elif len(equals) > 2:
            raise Exception('More than two duplicates.')
    return indices


def switch_left_and_right_labels(rec):
    left_labels = [
        'LFHD', 'LBHD', 'LSHO', 'LUPA', 'LELB', 'LFRM', 'LWRA', 'LWRB', 'LFIN', 
        'LASI', 'LPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE'
    ]
    right_labels = [
        'RFHD', 'RBHD', 'RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN',
        'RASI', 'RPSI', 'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE'
    ]
    left_indices = [data_info.joint_to_index[l] for l in left_labels]
    right_indices = [data_info.joint_to_index[r] for r in right_labels]

    new_left_data = rec[:, right_indices, :]
    new_right_data = rec[:, left_indices, :]

    rec[:, left_indices, :] = new_left_data
    rec[:, right_indices, :] = new_right_data

    return rec


def modify_data():
    new_data = copy.deepcopy(data)
    delete_idxs = []

    try:
        for i, (idx, length, reason) in enumerate(outliers):
            pre_mirror = False
            pre_switch = False
            rec = positions[idx]
            original_rec = copy.deepcopy(rec)
            start = 0
            end = int(length * frequency)
            done = False
            if str(idx) in report.keys():
                actions = copy.deepcopy(report[str(idx)]['actions'])
            else:
                actions = []
            report_step = {}
            report_step_done = False
            while not done:
                #print(original_rec[:, data_info.joint_to_index['STRN'], :])
                
                rec = copy.deepcopy(original_rec[start: end])

                #print(start, end)
                if ('mirror' in actions and str(idx) not in report.keys()) or pre_mirror:
                    rec[:, :, 1] *= -1
                    # Also need to switch sides because mirroring 
                    # would otherwise change the dominant hand. 
                    # No need to change left and right labels because
                    # mirrowing and switching cancel eachother out 
                    # in that regard. 
                    rec[:, :, 0] *= -1

                if ('switch side' in actions and str(idx) not in report.keys()) or pre_switch:  
                    rec[:, :, 0] *= -1
                    # Switch left and right labels
                    rec = switch_left_and_right_labels(rec)

                # Re-centering and calculating new joint angles and distances
                positions_df = pd.DataFrame(rec.reshape(-1, 39 * 3),
                        columns=data_prep.create_multi_index_cols(data_info.joint_to_index.keys(), time=False))
                
                #print(data[idx]['joint_positions'][:, data_info.joint_to_index['STRN'], :])
                #print(i, idx, length, reason)
                centered_positions_df = data_prep.center_initial_position(positions_df)

                rec = centered_positions_df.to_numpy().reshape(-1, 39, 3)
                rec.astype(object)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                rec_tensor = torch.as_tensor(rec, device=device)
                new_joint_axis_angles, new_joint_distances = geometry.calc_axis_angles_and_distances(
                    points=rec_tensor
                )

                # For testing if reconstructed motion is still correct
                '''
                start_label = 'T10'
                start_idx = data_info.joint_to_index[start_label]
                test_recon = geometry.calc_positions(
                    torch.tensor(rec[:, start_idx, :]).unsqueeze(dim=0), start_label,
                    new_joint_axis_angles.unsqueeze(dim=0), new_joint_distances.unsqueeze(dim=0))
                test_recon = test_recon.squeeze().cpu().detach().numpy()
                print('Testing reconstruction')
                from_array(test_recon)
                '''

                new_joint_axis_angles = new_joint_axis_angles.cpu().detach().numpy()
                new_joint_axis_angles.astype(object)
                new_joint_distances = new_joint_distances.cpu().detach().numpy()
                new_joint_distances.astype(object)

                if str(idx) in report.keys():
                    if report_step_done:
                        action = 'done'
                    else:
                        if len(actions) != 0:
                            action = actions[0]
                        else:
                            action = 'done'
                else:
                    if 'remove' in actions:
                        action = 'done'
                    else:
                        print(f'Length: {rec.shape[0] / frequency}')
                        print(f'Detection criteria: {reason}')
                        from_array(rec)
                        action = input('Choose an action (remove, mirror, trim, switch side) or type done for no further actions: ')

                if action == 'remove':
                    if str(idx) in report.keys():
                        report_step_done = True
                    else:
                        # When removing there is only one action
                        actions = [action]
                    delete_idxs.append(idx)
                elif action == 'mirror':
                    rec[:, :, 1] *= -1
                    # Also need to switch sides because mirroring 
                    # would otherwise change the dominant hand. 
                    rec[:, :, 0] *= -1
                    if str(idx) in report.keys():
                        actions.remove(action)
                        pre_mirror = True
                        if len(actions) == 0:
                            report_step_done = True
                    else:
                        if action in actions:
                            # Reversing the mirrowing in case
                            # it was unwanted. 
                            actions.remove(action)
                        else:
                            actions.append(action)
                elif action == 'switch side':
                    rec[:, :, 0] *= -1
                    # Switch left and right labels
                    rec = switch_left_and_right_labels(rec)
                    if str(idx) in report.keys():
                        actions.remove(action)
                        pre_switch = True
                        if len(actions) == 0:
                            report_step_done = True
                    else:
                        if action in actions:
                            # Reversing the switch in case
                            # it was unwanted. 
                            actions.remove(action)
                        else:
                            actions.append(action)
                elif action == 'trim':
                    if str(idx) in report.keys():
                        new_start_s, new_end_s = report[str(idx)]['borders']
                        new_start = int(new_start_s * frequency)
                        new_end = int(new_end_s * frequency)
                        actions.remove(action)
                        if len(actions) == 0:
                            report_step_done = True
                    else:
                        new_start_s, new_end_s = None, None
                        while new_start_s is None or new_end_s is None:
                            try:
                                new_start_s = float(input('New start (in seconds): '))
                                new_end_s = float(input('New end (in seconds): '))
                            except ValueError:
                                print('Invalid input. Please type in new frame borders.')
                                pass
                        new_start = int(new_start_s * frequency)
                        new_end = int(new_end_s * frequency)
                        new_end = min(new_end, int(length * frequency))
                        print(f'New frame borders: {new_start} and {new_end}')
                        if action not in actions:
                            actions.append(action)
                        report_step['borders'] = (new_start_s, new_end / frequency)
                    start = new_start
                    end = new_end
                elif action == 'done':
                    if str(idx) not in report.keys():
                        note = input('Pass a note (optional): ')
                        a = actions
                        report_step['actions'] = a
                        report_step['note'] = note
                        report_step['detection_criteria'] = reason
                        report[str(idx)] = report_step

                    if np.isnan(rec).any() or \
                            np.isnan(new_joint_axis_angles).any() or \
                            np.isnan(new_joint_distances).any():
                        print('Encountered nan value. Exiting...')
                        exit()

                    rec_data = copy.deepcopy(data[idx])
                    rec_data['joint_positions'] = rec
                    rec_data['joint_axis_angles'] = new_joint_axis_angles
                    rec_data['joint_distances'] = new_joint_distances
                    new_data[idx] = rec_data
                    done = True
                    print(f'Finished index {i} of the outliers (Total: {len(outliers)})')
                else:
                    if action == 'none' and str(idx) in report.keys():
                        done = True
                        print(f'Finished index {i} of the outliers.')
                        print('------')
                    else:
                        print('Wrong Input. This recording will be repeated.')
    except KeyboardInterrupt:
        print(f'\nAugmentation was interrupted at index {i} of the outliers.')
    finally:
        removed_duplicates_file_path = os.path.join(args.report_dir, "removed_duplicates.json")
        with open(removed_duplicates_file_path, 'w') as outfile:
            json.dump(duplicate_sample_indices, outfile)
        print(f'Saved a list of the removed duplicates at {removed_duplicates_file_path}.')

        delete_idxs.extend(duplicate_sample_indices)
        delete_idxs = list(set(delete_idxs))

        new_data = [x for (i, x) in enumerate(new_data) if i not in delete_idxs]
        new_data_file_path = os.path.join(args.target_dir, "karate_motion_modified.npy")
        np.save(new_data_file_path, new_data)
        print(f'Saved new data at {new_data_file_path}.')
        print(f'It contains {len(new_data)} recordings.')

        #print(f'Report:')
        #for key, value in report.items():
        #    print(f'{key}: {value}')

        with open(report_file_path, 'w') as outfile:
            json.dump(report, outfile)
        print(f'Saved report at {report_file_path}.')


def get_args():
    data_dir = os.path.join(os.getcwd(), 'datasets', 'karate')
    report_dir = os.path.join(os.getcwd(), 'preprocessing', 'karate', 'reports')

    parser = argparse.ArgumentParser(description='Modification of the karate motion data.')
    parser.add_argument('--data_dir', '-d', dest='data_dir', type=str, default=data_dir,
                        help='The directory that stores the unmodified karate npy data.')
    parser.add_argument('--report_dir', '-s', dest='report_dir', type=str, default=report_dir,
                        help='The directory that stores the report file.')
    parser.add_argument('--target_dir', '-t', dest='target_dir', type=str, default=data_dir,
                        help='The directory in which the modified data will be stored in.')
    parser.add_argument('--frequency', '-f', dest='frequency', type=int, default=25,
                        help='The data frequency (in Hz).')
    parser.add_argument('--replace', '-r', dest='replace', action='store_true', default=False,
                        help='Set if the existing data in the output directory should be replaced.')
    parser.add_argument('--use_report', '-u', dest='use_report', action='store_true', default=True,
                        help='Set a report should be used.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    target_data_file_path = os.path.join(args.target_dir, 'karate_motion_modified.npy')
    if os.path.isfile(target_data_file_path):
        if not args.replace:
            message = 'The target file already exists. If the data should be '
            message += 'replaced by new data, run this script with the --replace argument. Exiting...'
            raise Exception(message)

    data_file_path = os.path.join(args.data_dir, 'karate_motion_prepared.npy') #'karate_motion_unmodified.npy')
    data = np.load(data_file_path, allow_pickle=True)

    duplicate_sample_indices = find_duplicate_sample_indices()

    positions = [x for x in data["joint_positions"]]
    num_frames_in_video = [p.shape[0] for p in positions]
    frequency = args.frequency
    durations = np.array(num_frames_in_video) / frequency

    outliers = []

    # For checking that the modified dataset contains no outliers anymore
    '''
    #data_file_path = os.path.join(data_path, "karate_motion_modified.npy")
    #data_file_path = os.path.join(args.target_dir, "karate_motion_modified.npy")
    data_file_path = os.path.join(args.target_dir, "karate_motion_prepared.npy")
    #data_file_path = os.path.join(args.target_dir, "karate_motion_unmodified_before_switch.npy")    
    data = np.load(data_file_path, allow_pickle=True)
    new_positions = [x for x in data["joint_positions"]]
    num_frames_in_video = [p.shape[0] for p in new_positions]
    durations = np.array(num_frames_in_video) / args.frequency
    '''

    add_orientation_outliers()
    add_dominant_side_outliers()
    add_t_pose_outliers()
    add_duration_outliers()
    add_head_outliers()
    add_no_movement_outliers()
    add_manual_outliers()

    print(f'Total number of outliers: {len(outliers)}')
    

    use_report = args.use_report
    report_file_path = os.path.join(args.report_dir, "outlier_report.json")
    if use_report:
        if os.path.isfile(report_file_path):
            report = json.load(open(report_file_path))
        else:
            report = {}
    else:
        report = {}

    #print(len(report.keys()))
    #exit()
    
    '''c = 0
    for (idx, length, reason) in outliers:
        
        if str(idx) not in report.keys():
            c+= 1
            print(idx)
            p = positions[idx]
            #print(report[str(idx)])
            print(reason)
            #from_array(p)
        #else: 
        #    p = positions[idx]
        #    from_array(p)
    print(c)
    exit()'''

    # For removing already finished outliers from the 
    # report (to do them again). Make sure to 
    # backup the report before doing this. 
    '''number_of_removals = 0
    for (idx, length, reason) in outliers:
        if str(idx) in report.keys():
            #if len(reason.split(',')) > 0 and 'dominant' in reason and 'orientation' not in reason:
            report.pop(str(idx))

            print(reason)

            print(f'Removed idx: {idx}')
            number_of_removals += 1
    print(number_of_removals)
    with open(report_file_path, 'w') as outfile:
        json.dump(report, outfile)'''

    '''c = 0
    report_copy = report.copy()
    for idx in report_copy.keys():
        entry = report[idx]
        #if 'orientation' not in entry['detection_criteria'] and 'dominant' not in entry['detection_criteria']:
        if 'arning' in entry['note']:
            report.pop(idx)
            c += 1
            print(f'removed {idx}')
    print(c)
    with open(report_file_path, 'w') as outfile:
        json.dump(report, outfile)
    exit()'''

    '''number_of_removals = 0
    outlier_keys = [str(idx) for (idx, length, reason) in outliers]
    report_copy = report.copy()
    for k in report_copy.keys():
        if k not in outlier_keys:
    #for (idx, length, reason) in outliers:
    #    if str(idx) in report.keys():
            #if len(reason.split(',')) > 0 and 'dominant' in reason and 'orientation' not in reason:
            report.pop(str(k))

            #print(reason)

            print(f'Removed idx: {k}')
            number_of_removals += 1
    print(number_of_removals)
    with open(report_file_path, 'w') as outfile:
        json.dump(report, outfile)
    exit()'''

    print('------')
    print('Beginning modification')
    print('------')
    modify_data()
