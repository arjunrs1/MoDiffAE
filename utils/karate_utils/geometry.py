#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import utils.karate_utils.data_info as data_info
import vg
import utils.rotation_conversions as base_geometry
import torch

def points_to_axis_angle_and_distance(start_point, end_point):
    dist = np.linalg.norm(end_point - start_point)

    new_start = np.array([0., 0., 0.]) - start_point
    new_end = end_point - start_point
    new_start = new_start / np.linalg.norm(new_start)
    new_end = new_end / np.linalg.norm(new_end)

    axis = np.cross(new_start, new_end)
    axis = axis / np.linalg.norm(axis)
    signed_angle = vg.signed_angle(new_start, new_end, look=axis)
    signed_angle = np.radians(np.array([signed_angle]))[0]

    axis_angle = axis * signed_angle
    return axis_angle, dist


def calc_axis_angles_and_distances(joint_positions_df):
    skeleton = data_info.reconstruction_skeleton
    number_of_frames = joint_positions_df.shape[0]

    axis_angles = np.zeros(shape=(number_of_frames, len(skeleton) * 3))
    distances = np.zeros(shape=(len(skeleton),))
    for i, (s, t) in enumerate(skeleton):
        for f in range(number_of_frames):
            start_point = joint_positions_df[s].loc[f].to_numpy()
            end_point = joint_positions_df[t].loc[f].to_numpy()
            axis_angle, dist = points_to_axis_angle_and_distance(start_point, end_point)
            axis_angles[f, i*3:i*3+3] = axis_angle
            if f == 0:
                distances[i] = dist

    axis_angles.astype(object)
    return axis_angles, distances


def rodrigues_rotation(axes, angles, vectors):
    """
    Applies a Rodrigues rotation to a batch of vectors by an angle around the axis.
    Args:
        axes (torch.Tensor): A batch of 3D vectors representing the axes of rotation.
        angles (torch.Tensor): A batch of angles (in radians) by which to rotate around the axes.
        vectors (torch.Tensor): A batch of 3D vectors to be rotated.
    Returns:
        torch.Tensor: The batch of rotated vectors.
    """

    #print('shapes')
    #print(axes.shape)
    #print(angles.shape)
    #print(vectors.shape)

    # Make sure the batch sizes of axes, angles, and vectors match
    assert axes.shape[:-1] == angles.shape[:-1] == vectors.shape[:-1], "Batch dimensions of inputs do not match"

    # Broadcast angles along the last dimension of the vectors tensor
    angles = angles.expand_as(vectors)

    # Compute the tensor product between axes and vectors using broadcasting
    v_rot = torch.cross(axes, vectors, dim=-1)

    # Compute the dot product between axes and vectors
    # According to https://github.com/pytorch/pytorch/issues/18027 this is the most
    # efficient way of calculating the batch dot product.
    a_dot_v = torch.sum(axes * vectors, dim=-1, keepdim=True)

    # Compute the rotated vectors using Rodrigues' formula
    rotated_vectors = vectors * torch.cos(angles) + v_rot * torch.sin(angles) + axes * a_dot_v * (1 - torch.cos(angles))

    # Squeeze out the expanded batch dimensions and return the result
    return rotated_vectors.squeeze(-1)


def axis_angle_and_distance_to_point(start_points, axis_angles, distances):
    #new_start = np.array([0., 0., 0.]) - start_point
    # Centering coordinate system
    new_starts = torch.zeros_like(start_points) - start_points
    #new_starts = torch.zeros_like(start_points)   # - start_points

    # Normalizing
    norms = torch.linalg.norm(new_starts, dim=2)
    norms = norms.unsqueeze(dim=-1)
    new_starts = torch.div(new_starts, norms)
    #new_start = new_start / np.linalg.norm(new_start)

    #axis_angle = torch.tensor(axis_angle)

    angles = torch.linalg.norm(axis_angles, dim=2)
    angles = angles.unsqueeze(dim=-1)
    axes = torch.div(axis_angles, angles)

    #angles = angles.squeeze()

    # TODO: check if this is correct. I used chatgpt
    rec_end_points = rodrigues_rotation(axes, angles, new_starts)

    #print('scary')
    #print(rec_end_points.shape)

    #exit()

    ###
    #rot_matrix = base_geometry.axis_angle_to_matrix(axis_angles)
    #rec_end_point = np.dot(rot_matrix, new_start)
    distances = distances.unsqueeze(dim=-1)
    distances = distances.unsqueeze(dim=-1)
    #print(distances.shape)

    rec_end_points *= distances
    rec_end_points += start_points

    return rec_end_points


def calc_positions(chain_start_positions, start_label, axis_angles, distances):
    skeleton = data_info.reconstruction_skeleton
    if start_label != skeleton[0][0]:
        raise Exception('Start joint and skeleton order do not match!')

    print(axis_angles.shape)
    print(chain_start_positions.shape)
    print(distances.shape)
    #exit()

    nsamples, number_of_frames, njoints, feats = axis_angles.shape

    #number_of_frames = axis_angles.shape[0]
    #joint_positions = \
    #    np.zeros(shape=(number_of_frames, axis_angles.shape[1] + 3))
    joint_positions = \
        torch.zeros(size=(nsamples, number_of_frames, njoints + 1, 3))
    start_index = data_info.joint_to_index['T10']
    #joint_positions[:, start_index*3:start_index*3+3] = chain_start_positions
    #print(joint_positions)
    #print(torch.mean(joint_positions))
    joint_positions[:, :, start_index, :] = chain_start_positions
    #print(joint_positions)
    #print(torch.mean(joint_positions))

    #exit()

    #for f in range(number_of_frames):
    for i in range(len(skeleton)):
        start_label = skeleton[i][0]
        start_idx = data_info.joint_to_index[start_label]
        start_points = joint_positions[:, :, start_idx, :]

        #axis_angles = axis_angles[f, i * 3:i * 3 + 3]
        ax_angles = axis_angles[:, :, i, :]
        j_distances = distances[:, i]
        end_points = axis_angle_and_distance_to_point(start_points, ax_angles, j_distances)
        end_label = skeleton[i][1]
        end_idx = data_info.joint_to_index[end_label]
        #joint_positions[f, end_idx*3:end_idx*3+3] = end_point
        joint_positions[:, :, end_idx, :] = end_points

    return joint_positions
