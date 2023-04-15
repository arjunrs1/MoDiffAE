#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import data_info


def points_to_direction_and_distance(start_point, end_point):
    dir_vec = end_point - start_point
    dist = np.linalg.norm(dir_vec)
    dir_vec = dir_vec / dist
    return dir_vec, dist


def calc_joint_directions_and_distances(joint_positions_df):
    skeleton = data_info.reconstruction_skeleton
    number_of_frames = joint_positions_df.shape[0]

    directions = np.zeros(shape=(number_of_frames, len(skeleton) * 3))
    distances = np.zeros(shape=(len(skeleton),))
    for i, (s, t) in enumerate(skeleton):
        for f in range(number_of_frames):
            start_point = joint_positions_df[s].loc[f].to_numpy()
            end_point = joint_positions_df[t].loc[f].to_numpy()
            # a, d = points_to_direction_and_distance(start_point, end_point)
            dir_vec, dist = points_to_direction_and_distance(start_point, end_point)
            directions[f, i*3:i*3+3] = dir_vec
            if f == 0: 
                distances[i] = dist

    directions.astype(object)
    return directions, distances


def direction_and_distance_to_point(start_point, direction, distance):
    # Normalizing the direction vector because the network
    # does not necessarily output normalized direction vectors.
    # However, this is needed to maintain the correct joint distances.
    direction = direction / np.linalg.norm(direction)
    end_point = start_point + direction * distance
    return end_point


def calc_positions(chain_start_positions, start_label, directions, distances):
    skeleton = data_info.reconstruction_skeleton
    if start_label != skeleton[0][0]:
        raise Exception('Start joint and skeleton order do not match!')

    number_of_frames = directions.shape[0]
    joint_positions = \
        np.zeros(shape=(number_of_frames, directions.shape[1] + 3))
    joint_positions[:, 0:3] = chain_start_positions

    for f in range(number_of_frames):
        for i in range(len(skeleton)):
            if i == 0: 
                start_point = chain_start_positions[f]
            else: 
                start_label = skeleton[i][0]
                start_idx = data_info.joint_to_index[start_label]
                start_point = joint_positions[f, start_idx*3:start_idx*3+3]

            direction = directions[f, i * 3:i * 3 + 3]
            distance = distances[i]
            end_point = direction_and_distance_to_point(start_point, direction, distance)
            end_label = skeleton[i][1]
            end_idx = data_info.joint_to_index[end_label]
            joint_positions[f, end_idx*3:end_idx*3+3] = end_point

    return joint_positions
