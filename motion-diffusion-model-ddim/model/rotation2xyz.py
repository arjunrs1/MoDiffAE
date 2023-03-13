# This code is based on https://github.com/Mathux/ACTOR.git
import torch
import utils.rotation_conversions as geometry
import utils.karate_utils.geometry as karate_geometry
import numpy as np


from model.smpl import SMPL, JOINTSTYPE_ROOT
# from .get_model import JOINTSTYPES
JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices", "karate"] # added karate


class Rotation2xyz:
    def __init__(self, device, dataset='amass'):
        self.device = device
        self.dataset = dataset
        self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, get_rotations_back=False, distance=None, **kwargs):
        if pose_rep == "xyz":
            return x

        '''
        # TODO
        if jointstype == 'karate': 
            # - also pass distances in function as optional parameter DONE


            # - separate chain start position (and use only 3 values) 
            # and the rotations
            if translation:
                x_translations = x[:, -1, :3]
                x_rotations = x[:, :-1]
            else:
                x_rotations = x
            
            x_rotations = x_rotations.permute(0, 3, 1, 2)
            nsamples, time, njoints, feats = x_rotations.shape

            # - use the mask in similar fashion
            # - convert rotation6d to matrix (their geometry lib)
            # - convert matrix to axis angles (their geometry lib)
            # - use distances and axis angles to get back the 
            # xyz values (my geometry lib)

            # - for saftey check if the following returns the original values: 
            # axis angles to 6d, to matrix, to axis angle (using their lib and my data)

            # - also see what the rest is that they do here
            return x[:, :, :3, :]
        '''

        #print(mask)

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        #print(mask)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        #print(x_rotations.shape)
        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape
        #print(x_rotations.shape)
        #print(x_rotations[mask].shape)

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError("No geometry for this one.")

        if jointstype == "karate": 
            #print('worked!')
            #print(distance)

            #print(len(distance[0]))

            #print(x.shape)

            #print(len(distance))
            #print(distance[0].shape)

            rotations = geometry.matrix_to_axis_angle(rotations)

            #print(rotations.shape)

            rotations = torch.reshape(rotations, (nsamples, time, njoints, 3))

            #print(rotations.shape)
            #print(x_translations.shape)

            joints_list = []
            for i in range(nsamples):
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
                    start_label="LFHD", 
                    joint_angles=i_rotations, 
                    distances=i_distance
                )
                # It is important that the ordering is correct here.
                # Numpy reshape uses C like indexing by default.
                i_joints = np.reshape(i_joints, (time, njoints+1, 3))
                joints_list.append(i_joints)

            joints = torch.tensor(joints_list, device=x.device, dtype=x.dtype)  
            joints = torch.reshape(joints, (nsamples*time, njoints+1, 3))

        else: 
            if not glob:
                global_orient = torch.tensor(glob_rot, device=x.device)
                global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
            else:
                global_orient = rotations[:, 0]
                rotations = rotations[:, 1:]

            if betas is None:
                betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                    dtype=rotations.dtype, device=rotations.device)
                betas[:, 1] = beta
                # import ipdb; ipdb.set_trace()
            out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

            # get the desirable joints
            joints = out[jointstype]
            #print(joints.shape)

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

         # Do not do this on karate data. The root joint is not centered.
        if jointstype != "karate":
            # the first translation root at the origin on the prediction
            if jointstype != "vertices":
                rootindex = JOINTSTYPE_ROOT[jointstype]
                x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

            if translation and vertstrans:
                # the first translation root at the origin
                x_translations = x_translations - x_translations[:, :, [0]]

                # add the translation to all the joints
                x_xyz = x_xyz + x_translations[:, None, :, :]

        if get_rotations_back:
            if jointstype == "karate":
                raise NotImplementedError("This global orientation is not implemented for the karate joints type.")
            return x_xyz, rotations, global_orient
        else:
            return x_xyz
