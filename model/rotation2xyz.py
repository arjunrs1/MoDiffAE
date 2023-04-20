# This code is based on https://github.com/Mathux/ACTOR.git
import torch
import utils.rotation_conversions as geometry
import utils.karate_utils.geometry as karate_geometry
import numpy as np

JOINT_TYPES = ["karate"]


class Rotation2xyz:
    def __init__(self, device):  # , dataset='amass'):
        self.device = device
        # self.dataset = dataset

    def __call__(self, x, mask, pose_rep, translation,
                 joint_type='karate', distance=None, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if joint_type not in JOINT_TYPES:
            raise NotImplementedError("This joint type is not implemented.")

        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        #print(x.shape)
        #print(x_rotations.shape)
        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape
        #print(x_rotations.shape)
        #print(x_rotations[mask].shape)

        #exit()

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rot_vec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rot_mat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rot_quat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError("No geometry for this pose representation.")

        rotations = geometry.matrix_to_axis_angle(rotations)
        rotations = torch.reshape(rotations, (nsamples, time, njoints, 3))

        #num = np.random.randn()
        #print(f'{num} origin {rotations}')
        #ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(rotations))
        #rec_pose = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(ret))
        #print(f'{num} recon {rec_pose}')

        #exit()

        joints_list = []
        for i in range(nsamples):
            print(i)
            i_start_positions = x_translations[i].permute(1, 0).cpu().detach().numpy()
            #i_rotations = torch.reshape(rotations[i], (time, njoints*3))
            i_rotations = rotations[i]
            i_rotations = i_rotations.cpu().detach().numpy()
            # It is important that the ordering is correct here.
            # Numpy reshape uses C like indexing by default.
            i_rotations = np.reshape(i_rotations, (time, njoints*3))
            i_distance = distance[i]
            i_joints = karate_geometry.calc_positions(
                chain_start_positions=i_start_positions,
                start_label='T10',   # "LFHD",
                axis_angles=i_rotations,
                distances=i_distance
            )
            # It is important that the ordering is correct here.
            # Numpy reshape uses C like indexing by default.
            i_joints = np.reshape(i_joints, (time, njoints+1, 3))
            joints_list.append(i_joints)

        joints = torch.tensor(joints_list, device=x.device, dtype=x.dtype)
        joints = torch.reshape(joints, (nsamples*time, njoints+1, 3))

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        return x_xyz
