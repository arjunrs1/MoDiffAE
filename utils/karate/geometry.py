#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils.karate.data_info as data_info
import torch

# This geometry is implemented so that is works both on the gpu and the cpu.


def calc_signed_axis_angle(start, end):
    v_cross = torch.cross(start, end, dim=-1)
    v_dot = torch.sum(start * end, dim=-1, keepdim=True)
    axis = torch.div(v_cross, torch.linalg.norm(v_cross, dim=-1).unsqueeze(dim=-1))
    # The determinant of the 3x3 matrix consisting of three vectors is defined as dot((cross(start, end), axis).
    # https://www.math.umd.edu/~petersd/241/crossprod.pdf
    det = torch.sum(v_cross * axis, dim=-1, keepdim=True)
    angle = torch.atan2(det, v_dot)
    axis_angle = axis * angle
    return axis_angle


def points_to_axis_angles_and_distances(start_points, end_points):
    dist = torch.linalg.norm(end_points - start_points, dim=-1)
    new_starts = torch.zeros_like(start_points) - start_points
    new_ends = end_points - start_points

    new_starts = torch.div(new_starts, torch.linalg.norm(new_starts, dim=-1).unsqueeze(dim=-1))
    new_ends = torch.div(new_ends, torch.linalg.norm(new_ends, dim=-1).unsqueeze(dim=-1))

    axis_angle = calc_signed_axis_angle(new_starts, new_ends)
    return axis_angle, dist


def calc_axis_angles_and_distances(points):
    # This is implemented so that it works for one complete sequence.
    # For multiple reasons this is more practical than creating a mask and processing multiple movements at once.
    # Especially in the modification we need to perform geometry on single movements.
    skeleton = data_info.reconstruction_skeleton
    n_frames = points.shape[0]

    axis_angles = torch.zeros(size=(n_frames, len(skeleton), 3), device=points.device)
    distances = torch.zeros(size=(len(skeleton),), device=points.device)
    for i, (s, t) in enumerate(skeleton):
        start_idx = data_info.joint_to_index[s]
        end_idx = data_info.joint_to_index[t]

        start_points = points[:, start_idx, :]
        end_points = points[:, end_idx, :]
        angles, dist = points_to_axis_angles_and_distances(start_points, end_points)

        dist = torch.mean(dist, dim=-1)
        distances[i] = dist
        axis_angles[:, i, :] = angles

    print(axis_angles.device, distances.device)

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
    # Centering coordinate system at old start point and defining new start point
    # Note that this automatically copies the device type
    new_starts = torch.zeros_like(start_points) - start_points

    # Normalizing
    norms = torch.linalg.norm(new_starts, dim=-1)
    norms = norms.unsqueeze(dim=-1)
    new_starts = torch.div(new_starts, norms)

    angles = torch.linalg.norm(axis_angles, dim=-1)
    angles = angles.unsqueeze(dim=-1)
    axes = torch.div(axis_angles, angles)

    rec_end_points = rodrigues_rotation(axes, angles, new_starts)

    distances = distances.unsqueeze(dim=-1).unsqueeze(dim=-1)

    rec_end_points *= distances
    rec_end_points += start_points

    return rec_end_points


def calc_positions(chain_start_positions, start_label, axis_angles, distances):
    skeleton = data_info.reconstruction_skeleton
    if start_label != skeleton[0][0]:
        raise Exception('Start joint and skeleton order do not match!')

    n_samples, n_frames, n_joints, n_feats = axis_angles.shape

    joint_positions = \
        torch.zeros(size=(n_samples, n_frames, n_joints + 1, 3), device=axis_angles.device)

    start_index = data_info.joint_to_index['T10']
    joint_positions[:, :, start_index, :] = chain_start_positions

    for i in range(len(skeleton)):
        start_label = skeleton[i][0]
        start_idx = data_info.joint_to_index[start_label]
        start_points = joint_positions[:, :, start_idx, :]

        ax_angles = axis_angles[:, :, i, :]
        j_distances = distances[:, i]
        end_points = axis_angle_and_distance_to_point(start_points, ax_angles, j_distances)
        end_label = skeleton[i][1]
        end_idx = data_info.joint_to_index[end_label]
        joint_positions[:, :, end_idx, :] = end_points

    print(joint_positions.device)

    return joint_positions
