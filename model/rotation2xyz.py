import torch
import utils.rotation_conversions as geometry
import utils.karate.geometry as karate_geometry

DATA_NAMES = ["karate"]


class Rotation2xyz:
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, x, mask, pose_rep, translation, data_name='karate',
                 distance=None, **kwargs):

        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=torch.bool, device=x.device)

        if data_name not in DATA_NAMES:
            raise NotImplementedError("This joint type is not implemented.")

        x_translations = None
        if translation:
            x_translations = x[:, -1, :3]
            x_translations = x_translations.permute(0, 2, 1)
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        # The transformer expects the last dimension to be the time steps.
        # For the conversion we want the rotation values in the last dimension.
        x_rotations = x_rotations.permute(0, 3, 1, 2)
        n_samples, n_time_steps, n_joints, n_feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rot_vec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rot_mat":
            rotations = x_rotations[mask].view(-1, n_joints, 3, 3)
        elif pose_rep == "rot_quat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot_6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError("No geometry for this pose representation.")

        if data_name == 'karate':
            rotations = geometry.matrix_to_axis_angle(rotations)
            rotations = torch.reshape(rotations, (n_samples, n_time_steps, n_joints, 3))
            joints = karate_geometry.calc_positions(
                chain_start_positions=x_translations,
                start_label='T10',
                axis_angles=rotations,
                distances=distance
            )
            joints = torch.reshape(joints, (n_samples * n_time_steps, n_joints + 1, 3))
        else:
            raise NotImplementedError("No conversion implemented for this joint type.")

        x_xyz = torch.empty(n_samples, n_time_steps, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        return x_xyz
