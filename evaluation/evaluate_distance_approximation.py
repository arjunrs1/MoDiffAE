#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

import torch

from visualize.vicon_visualization import from_array
import utils.karate.data_info as data_info
from load.data_loaders.karate import KaratePoses
from model.rotation2xyz import Rotation2xyz
import utils.rotation_conversions as geometry
import utils.karate.geometry as karate_geometry


###########
# Standard deviation of joint distances

data_dir = os.path.join(os.getcwd(), 'datasets', 'karate', 'karate_motion_prepared.npy')
data = np.load(data_dir, allow_pickle=True)

joint_distance_variations = []

for i, sample in enumerate(data):
    d = sample['joint_positions']

    for link in data_info.reconstruction_skeleton:

        link_indices = [data_info.joint_to_index[j] for j in link]
        start_positions = d[:, link_indices[0], :]
        end_positions = d[:, link_indices[1], :]

        distances = np.linalg.norm(end_positions - start_positions, axis=1)
        distance_std = np.std(distances)

        if distance_std > 50:
            print(link, distance_std, i)
            print(f'Ignoring the value')
            from_array(d, mode='inspection')
        else:
        # It was found that all the times (8) it was over 50 were marker errors.
        #if distance_std <= 50:
            joint_distance_variations.append(distance_std)


avg_joint_std = np.mean(joint_distance_variations)
max_joint_std = np.max(joint_distance_variations)
print(f'Average std: {avg_joint_std}')
print(f'Maximum std: {max_joint_std}')

print('####')

###########
# Average distance reconstruction error

'''test_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='test'
    )'''

# iterate over each one, put into tensor, expand and convert to 6d and then back

rot2xyz = Rotation2xyz(device='cpu')

distances_errors_list = []
for i, sample in enumerate(data):
    #print(i)
    sample_xyz = sample['joint_positions']

    dist = sample['joint_distances']
    dist = torch.as_tensor(dist)

    sample_axis_angles = sample['joint_axis_angles']
    sample_axis_angles = torch.as_tensor(sample_axis_angles)

    #print(sample_axis_angles.shape, dist.shape)

    sample_rot_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(sample_axis_angles))
    #print(sample_rot_6d.shape)
    sample_axis_angles = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(sample_rot_6d))
    #print(sample_axis_angles.shape)

    root_joint_idx = data_info.joint_to_index['T10']
    chain_starts = sample_xyz[:, root_joint_idx, :]

    chain_starts = torch.as_tensor(chain_starts).unsqueeze(0)
    sample_axis_angles = sample_axis_angles.unsqueeze(0)
    dist = dist.unsqueeze(0)

    reconstructed_xyz = karate_geometry.calc_positions(
        chain_start_positions=chain_starts,
        start_label='T10',
        axis_angles=sample_axis_angles,
        distances=dist
    )
    reconstructed_xyz = reconstructed_xyz.squeeze()
    reconstructed_xyz = reconstructed_xyz.numpy()

    #print(reconstructed_xyz.shape, sample_xyz.shape)

    distance_errors = np.linalg.norm(sample_xyz - reconstructed_xyz, axis=-1)

    distances_list = distance_errors.flatten().tolist()

    if max(distances_list) > 200:
        print(i)

    distances_list = [di for di in distances_list if di < 200]

    #print(distance_errors.shape)

    distances_errors_list.extend(distances_list)

    #print(np.mean(distance_errors))

print(np.mean(distances_errors_list))


'''sample_rot_6d = sample_rot_6d.unsqueeze(dim=0)

print(sample_rot_6d.shape)

# back to xyz
reconstructed_xyz = rot2xyz(sample_rot_6d, mask=None, pose_rep='rot_6d', translation=True,
                            data_name='karate', distance=dist)

print(reconstructed_xyz.shape)


exit()'''