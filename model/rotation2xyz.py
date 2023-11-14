import torch
import utils.rotation_conversions as geometry
import utils.karate.geometry as karate_geometry

from model.smpl import SMPL, JOINTSTYPE_ROOT

DATA_NAMES = ["karate", "humanact12"]


class Rotation2xyz:
    def __init__(self, device='cpu'):
        self.device = device
        #self.dataset = dataset
        #self.smpl_model = SMPL().eval().to(device)
    def __call__(self, x, mask, pose_rep, translation, data_name='karate',
                 distance=None, **kwargs):

        if pose_rep == "xyz":
            #print("hi")
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

        #print(x_rotations.shape)

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

        #rotations = geometry.matrix_to_axis_angle(rotations)
        #print(f'sh: {rotations.shape}')
        #rotations = torch.reshape(rotations, (n_samples, n_time_steps, n_joints, 3))

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
        elif data_name == 'humanact12':
            betas = None
            beta = 0
            glob = True
            glob_rot = None
            #print('TODO')
            if not glob:
                global_orient = torch.tensor(glob_rot, device=x.device)
                global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
            else:
                global_orient = rotations[:, 0]
                #print(global_orient.shape)

                rotations = rotations[:, 1:]

                #print(rotations.shape)

            if betas is None:
                betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                    dtype=rotations.dtype, device=rotations.device)
                betas[:, 1] = beta
                # import ipdb; ipdb.set_trace()

            #print(rotations.shape)
            #exit()
            self.smpl_model = self.smpl_model.to(device=self.device)
            rotations = rotations.to(device=self.device)
            global_orient = global_orient.to(device=self.device)
            betas = betas.to(device=self.device)
            out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

            # get the desirable joints
            #joints = out[jointstype]
            joints = out['smpl']

            joints = joints.to(device=x.device)
            # print(joints.shape)
        else:
            raise NotImplementedError("No conversion implemented for this joint type.")

        x_xyz = torch.empty(n_samples, n_time_steps, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if data_name == "humanact12":
            root_index = JOINTSTYPE_ROOT['smpl']
            x_xyz = x_xyz - x_xyz[:, [root_index], :, :]

        return x_xyz
