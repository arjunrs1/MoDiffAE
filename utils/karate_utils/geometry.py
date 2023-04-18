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


def axis_angle_and_distance_to_point(start_point, axis_angle, distance):
    # Normalizing the direction vector because the network
    # does not necessarily output normalized direction vectors.
    # However, this is needed to maintain the correct joint distances.
    #direction = direction / np.linalg.norm(direction)
    #end_point = start_point + direction * distance
    #return end_point

    new_start = np.array([0., 0., 0.]) - start_point
    new_start = new_start / np.linalg.norm(new_start)

    axis_angle = torch.tensor(axis_angle)
    rot_matrix = base_geometry.axis_angle_to_matrix(axis_angle)
    #rot_matrix = axis_angle_to_matrix(axis_angle)
    rec_end_point = np.dot(rot_matrix, new_start)
    rec_end_point *= distance
    rec_end_point += start_point

    return rec_end_point



def calc_positions(chain_start_positions, start_label, axis_angles, distances):
    skeleton = data_info.reconstruction_skeleton
    if start_label != skeleton[0][0]:
        raise Exception('Start joint and skeleton order do not match!')

    number_of_frames = axis_angles.shape[0]
    joint_positions = \
        np.zeros(shape=(number_of_frames, axis_angles.shape[1] + 3))
    start_index = data_info.joint_to_index['T10']
    joint_positions[:, start_index*3:start_index*3+3] = chain_start_positions
    #joint_positions[:, 0:3] = chain_start_positions

    for f in range(number_of_frames):
        for i in range(len(skeleton)):
            #if i == 0:
            #    start_point = chain_start_positions[f]
            #else:
            start_label = skeleton[i][0]
            start_idx = data_info.joint_to_index[start_label]
            start_point = joint_positions[f, start_idx*3:start_idx*3+3]

            axis_angle = axis_angles[f, i * 3:i * 3 + 3]
            distance = distances[i]
            end_point = axis_angle_and_distance_to_point(start_point, axis_angle, distance)
            end_label = skeleton[i][1]
            end_idx = data_info.joint_to_index[end_label]
            joint_positions[f, end_idx*3:end_idx*3+3] = end_point

    return joint_positions

##########

'''

def points_to_angles_and_distance(start_point, end_point):
    vec = end_point - start_point
    distance = np.linalg.norm(vec)
    angles = np.arccos(np.true_divide(vec, distance))
    return angles, distance

def calc_joint_angles_and_distances(joint_positions_df):
    skeleton = data_info.reconstruction_skeleton
    number_of_frames = joint_positions_df.shape[0]
    joint_angles = np.zeros(shape=(number_of_frames, len(skeleton) * 3))

    distances = np.zeros(shape=(len(skeleton),))
    for i, (s, t) in enumerate(skeleton):
        for f in range(number_of_frames):
            start_point = joint_positions_df[s].loc[f].to_numpy()
            end_point = joint_positions_df[t].loc[f].to_numpy()
            a, d = points_to_angles_and_distance(start_point, end_point)
            joint_angles[f, i*3:i*3+3] = a
            if f == 0: 
                distances[i] = d

    joint_angles.astype(object)
    return joint_angles, distances

def angles_and_distance_to_point(start_point, angles, distance): 
    end_point = start_point + np.cos(angles) * distance
    return end_point

def calc_positions(chain_start_positions, start_label, joint_angles, distances):
    skeleton = data_info.reconstruction_skeleton
    if start_label != skeleton[0][0]:
        raise Exception('Start joint and skeleton order do not match!')

    number_of_frames = joint_angles.shape[0]
    joint_positions = \
        np.zeros(shape=(number_of_frames, joint_angles.shape[1] + 3))
    joint_positions[:, 0:3] = chain_start_positions

    for f in range(number_of_frames):
        for i in range(len(skeleton)):
            if i == 0: 
                start_point = chain_start_positions[f]
            else: 
                start_label = skeleton[i][0]
                start_idx = data_info.joint_to_index[start_label]
                start_point = joint_positions[f, start_idx*3:start_idx*3+3]

            a = joint_angles[f, i*3:i*3+3]
            d = distances[i]

            end_point = angles_and_distance_to_point(start_point, a, d)
            end_label = skeleton[i][1]
            end_idx = data_info.joint_to_index[end_label]

            joint_positions[f, end_idx*3:end_idx*3+3] = end_point

    return joint_positions
'''