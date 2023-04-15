import pickle as pkl
import numpy as np
import os
from .dataset import Dataset
from utils.karate_utils.mocap_visualization import from_array
from data_loaders.get_data import get_dataset_loader
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
import torch
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from utils.fixseed import fixseed

import utils.rotation_conversions as geometry

import vg

from scipy.spatial.transform import Rotation as R

# import matplotlib.pyplot as plt
# import scipy.stats as stats
# import seaborn as sns
# import matplotlib.lines as lines


class KaratePoses(Dataset):
    dataname = "karate"

    def __init__(self, datapath="datasets/KaratePoses", split="train", **kargs):
        self.datapath = datapath

        # max num_frames in karate dataset is 356 (almost 15 seconds with 25 fps). 
        # But this seems too long. Check the recordings with long lengths.
        # num_frames=-2, min_len=2, max_len=356,
        # num_frames=125
        super().__init__(**kargs)

        self.data_name = "karate"

        npydatafilepath = os.path.join(datapath, "karate_motion_25_fps.npy")
        data = np.load(npydatafilepath, allow_pickle=True)

        self._data = data

        self._pose = [x for x in data["joint_angles"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]

        self._joints = [x for x in data["joint_positions"]]

        self._actions = [x for x in data["technique_cls"]]

        self._joint_distances = [x for x in data["joint_distances"]]

        total_num_actions = 5
        self.num_actions = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = karate_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix].reshape(-1, 39, 3)

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 38, 3)
        return pose


karate_action_enumerator = {
    0: 'Gyaku-Zuki',
    1: 'Mae-Geri',
    2: 'Mawashi-Geri gedan',
    3: 'Mawashi-Geri jodan',
    4: 'Ushiro-Mawashi-Geri'
}


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":

    p_wrong = torch.tensor([3.0405, 1.6361, 1.6478], dtype=torch.float64)
    p_wrong_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(p_wrong))
    p_wrong_rec = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(p_wrong_6d))
    #print(p_wrong)
    #print(p_wrong_rec)

    p_correct = torch.tensor([-1.9529, -1.0509, -1.0584], dtype=torch.float64)
    p_correct_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(p_correct))
    p_correct_rec = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(p_correct_6d))
    #print(p_correct)
    #print(p_correct_rec)

    p_test = torch.tensor([-0.15823568, 0.31647137, -0.15823568], dtype=torch.float64)
    p_test_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(p_test))
    p_test_rec = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(p_test_6d))
    #print(p_test)
    #print(p_test_rec)

    p_test2 = torch.tensor([0.6883753, 0.72296702, 0.05880593], dtype=torch.float64)
    p_test2_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(p_test2))
    p_test2_rec = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(p_test2_6d))
    #print(p_test2)
    #print(p_test2_rec)

    #exit()

    ####

    #start_point = np.array([1., 2., 3.])
    #end_point = np.array([200., 211., 20.])
    end_point = np.array([1., 2., 3.])
    start_point = np.array([-300., -211., -20.])
    point_dist = np.linalg.norm(end_point - start_point) # saved
    #st = start_point
    #en = end_point

    # Centering system at the angle between the two sides
    new_start = np.array([0., 0., 0.]) - start_point
    new_end = end_point - start_point
    print(new_start, new_end)
    # point_dist = np.linalg.norm(new_end - new_start)
    print(point_dist)
    # len_new_start = np.linalg.norm(new_start)   # can be calculated

    new_start = new_start / np.linalg.norm(new_start)
    new_end = new_end / np.linalg.norm(new_end)
    #angle = np.arccos(np.dot(start_point, end_point))
    #print(angle)
    #axis_old = np.cross(start_point, end_point)
    #axis_old = axis_old / np.linalg.norm(axis_old)
    axis = np.cross(new_start, new_end)
    axis = axis / np.linalg.norm(axis)
    #print(axis_old)
    print(axis)

    signed_angle = vg.signed_angle(new_start, new_end, look=axis)
    print(signed_angle)

    signed_angle = np.radians(np.array([signed_angle]))[0]
    print(signed_angle)

    print(axis)
    axis = axis * signed_angle   # model input
    print(axis)
    print('##')
    # reconstruct angle and axis from this
    # the len between the known and the unknown joint is stored
    # transform the new start (normalized) in the direction of the new end
    # scale that vector to the stored length. That should then be the new_end position
    # finally add the new start again to get the original coordinate system
    axis = torch.tensor(axis)

    rot_matrix = geometry.axis_angle_to_matrix(axis)
    # new_start is already normalized
    rec_end_point = np.dot(rot_matrix, new_start)
    print(rec_end_point)


    #l_rotated_start = np.linalg.norm(rec_end_point)
    #l_desired = dist
    #scaler = l_desired / l_rotated_start

    #print(scaler)

    rec_end_point *= point_dist
    print(rec_end_point)
    rec_end_point += start_point
    print(rec_end_point)



    exit()
    #####




    start_point = start_point / np.linalg.norm(start_point)
    end_point = end_point / np.linalg.norm(end_point)
    angle = np.arccos(np.dot(start_point, end_point))
    print(angle)
    axis = np.cross(start_point, end_point)
    axis = axis / np.linalg.norm(axis)

    #signed_angle = vg.signed_angle(start_point, end_point, look=axis)
    signed_angle = vg.signed_angle(end_point, start_point, look=axis)
    print(np.radians(np.array([signed_angle])))
    print(signed_angle)

    # TODO: if negative, simply add 2 pi (?)


    print(np.arccos(np.dot(start_point, end_point)))
    print(np.arccos(np.dot(end_point, start_point)))
    exit()

    #dist = np.linalg.norm(end_point - start_point)
    dist = np.linalg.norm(end_point)

    start_point = start_point / np.linalg.norm(start_point)
    end_point = end_point / np.linalg.norm(end_point)
    angle = np.arccos(np.dot(start_point, end_point))
    axis = np.cross(start_point, end_point)
    axis = axis / np.linalg.norm(axis)
    axis = axis * angle
    axis = torch.tensor(axis)

    rot_matrix = geometry.axis_angle_to_matrix(axis)

    rec_end_point = np.dot(rot_matrix, start_point)
    print(rec_end_point)

    print()



    l_rotated_start = np.linalg.norm(rec_end_point)
    l_desired = dist
    scaler = l_desired / l_rotated_start

    print(scaler)

    rec_end_point *= scaler
    print(rec_end_point)

    exit()


    #rec_angle = np.linalg.norm(axis)
    #rec_axis = axis / rec_angle

    #rec_end_point = np.multiply(np.cos(rec_angle), start_point) + np.multiply(np.sin(rec_angle), (
    #    np.cross(rec_axis, start_point))) + np.multiply(
    #    np.multiply((1 - np.cos(rec_angle)), (np.dot(rec_axis, start_point))), rec_axis)

    rec_end_point = np.cos(rec_angle) * start_point + np.sin(rec_angle) * (np.cross(rec_axis, start_point)) + (1 - np.cos(rec_angle)) # * (np.dot(rec_axis, start_point)) * rec_axis
    print(rec_end_point)


    exit()


    #####

    start_point = np.array([1, 2, 3])
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device='cpu')
    identity_matrix = geometry.quaternion_to_matrix(identity_quat)
    #print(identity_matrix)
    end_point = np.array([200, 211, 20])

    transformed = np.dot(identity_matrix, end_point)

    # usually the start and end would be transformed too but here it is the identity matrix
    print(np.linalg.norm(end_point - start_point))
    start_point = start_point / np.linalg.norm(start_point)
    end_point = end_point / np.linalg.norm(end_point)
    angle = np.arccos(np.dot(start_point, end_point))
    axis = np.cross(start_point, end_point)
    axis = axis / np.linalg.norm(axis)
    axis = axis * angle
    axis = torch.tensor(axis)
    axis_quat = geometry.axis_angle_to_quaternion(axis)
    #print(identity_quat)
    print(axis_quat)
    print(geometry.quaternion_to_axis_angle(axis_quat))


    quat = geometry.quaternion_raw_multiply(identity_quat, axis_quat)
    #quat = ge identity_quat * axis_quat
    #print(end_point)
    #print(transformed)
    #print(quat)

    quat_matrix = geometry.quaternion_to_matrix(quat)
    next_point = np.array([100, 11, 2])
    tr_previous = np.dot(quat_matrix, transformed)
    tr_next = np.dot(quat_matrix, next_point)
    #print(tr_previous)
    #print(tr_next)
    #print(np.linalg.norm(tr_next - tr_previous))
    #print(np.linalg.norm(np.array([100, 11, 2]) - np.array([200, 211, 20])))

    # axis angle between transformed and tr_next (in new coordinate system)
    print(np.linalg.norm(tr_next - tr_previous))
    tr_previous = tr_previous / np.linalg.norm(tr_previous)
    tr_next = tr_next / np.linalg.norm(tr_next)
    angle2 = np.arccos(np.dot(tr_previous, tr_next))
    axis2 = np.cross(tr_previous, tr_next)
    axis2 = axis2 / np.linalg.norm(axis2)
    axis2 = axis2 * angle2
    axis2 = torch.tensor(axis2)
    axis_quat2 = geometry.axis_angle_to_quaternion(axis2)
    print(axis_quat2)
    print(geometry.quaternion_to_axis_angle(axis_quat2))


    quat = geometry.quaternion_raw_multiply(quat, axis_quat2)

    ##
    # used points: np.array([1, 2, 3]), np.array([200, 211, 20]), np.array([100, 11, 2])
    # extracted (relative) axis angles:
    # quat: tensor([ 0.9015, -0.3016,  0.2950, -0.0961], dtype=torch.float64), as axis angle: tensor([-0.6239,  0.6102, -0.1988], dtype=torch.float64), distance: 289.0864922475625
    # quat: tensor([ 0.9388, -0.1997, -0.1453, -0.2403], dtype=torch.float64), as axis angle: tensor([-0.4077, -0.2966, -0.4906], dtype=torch.float64), distance: 224.33011389467978
    # TODO: reconstruct the two other points from the start point [1, 2, 3] using the axis angles. Starting at the identity quaternion
    ##

    start = np.array([1, 2, 3])
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device='cpu')
    r = R.from_rotvec(torch.tensor([-0.6239,  0.6102, -0.1988], dtype=torch.float64))
    yaw, pitch, roll = r.as_euler('zxy').tolist() # what is the order? yaw, pitch, roll ?
    #angles_test = r.as_euler('zxy').tolist()  # what is the order? yaw, pitch, roll ?
    #angles_test = [angles_test[1], angles_test[2], angles_test[0]]
    print(yaw, pitch, roll)
    x = np.cos(yaw) * np.cos(pitch)
    y = np.sin(yaw) * np.cos(pitch)
    z = np.sin(pitch)
    dir_vec = np.array([x, y, z])
    print(dir_vec)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    print(dir_vec)
    #end_test = start + np.cos(angles_test) * 289.0864922475625
    end_test = start + dir_vec * 289.0864922475625
    print(end_test)




    exit()

    ####

    start_point = np.array([1, 2, 3])
    og_start = start_point
    end_point = np.array([200, 211, 20])
    # dist = np.linalg.norm(end_point - start_point)

    #start_point = start_point / np.linalg.norm(start_point)
    #end_point = end_point / np.linalg.norm(end_point)
    # print()
    #vec = end_point - start_point
    #angle = np.arccos(np.dot(start_point, end_point))  # / (np.linalg.norm(vec)))

    axis = end_point - start_point # np.cross(start_point, end_point)
    dist = np.linalg.norm(end_point - start_point)
    axis = axis / np.linalg.norm(end_point - start_point)

    #axis = axis * dist

    #print(angle)
    print(axis)
    print(dist)
    print(np.linalg.norm(axis))
    print(og_start + (axis * dist))

    exit()

    ####


    start_point = np.array([1, 2, 3])
    og_start = start_point
    end_point = np.array([2, 2, 2])
    dist = np.linalg.norm(end_point - start_point)

    start_point = start_point / np.linalg.norm(start_point)
    end_point = end_point / np.linalg.norm(end_point)
    #print()
    #vec = end_point - start_point
    angle = np.arccos(np.dot(start_point, end_point)) #/ (np.linalg.norm(vec)))

    axis = np.cross(start_point, end_point)
    axis = axis / np.linalg.norm(axis)



    axis = axis * angle

    print(angle)
    print(axis)
    print(dist)
    print(np.linalg.norm(axis))
    print(og_start + (axis * dist))

    exit()

    vec = vec / np.linalg.norm(vec)
    print(angle)
    ax_angle_3d = vec * angle

    print(ax_angle_3d)

    exit()

    vec = end_point - start_point
    dot_prod = np.dot(start_point, end_point)
    print(vec, dot_prod)
    distance = np.linalg.norm(vec)
    angles = np.arccos(np.true_divide(vec, distance))
    print(angles, distance)

    exit()

    kp = KaratePoses()
    kp.num_frames = 125
    # print(kp._pose[1].shape)
    t = kp._load_rotvec(0, 0)
    # print(kp._load_joints3D(0, 0))
    # print(t)

    # print('max: ' + str(max(kp._num_frames_in_video)))
    # print('max: ' + str(np.mean(kp._num_frames_in_video)))
    # print(kp._num_frames_in_video)

    # print(kp._num_frames_in_video[0])

    # loaded = kp[20]
    # print(loaded)
    # print(loaded['inp'].shape)
    # print()

    d = kp._data['joint_positions'][0]
    print(d.shape)
    from_array(d)

    # d_post = kp[0]
    # print(d_post['inp'].shape)

    ################
    args = generate_args()
    fixseed(args.seed)
    max_frames = 125
    fps = 25
    n_frames = min(max_frames, int(args.motion_length * fps))

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    # if is_using_data:
    iterator = iter(data)
    # model_kwargs contains the condition as well as distances
    # for visualization of karate motion.
    data_batch, model_kwargs = next(iterator)
    '''
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = collate(collate_args)
    '''

    #rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
    rot2xyz_pose_rep = model.data_rep
    # rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(1, n_frames).bool()
    rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size,
                                                                                            n_frames).bool()

    distance = model_kwargs['y']['distance']
    og_xyz = model.rot2xyz(x=data_batch, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, translation=True,
                           distance=distance)

    og_xyz = og_xyz.cpu().numpy()

    lengths = model_kwargs['y']['lengths'].cpu().numpy()

    length = lengths[0]
    m = og_xyz[0].transpose(2, 0, 1)[:length]

    t, j, ax = m.shape

    # It is important that the ordering is correct here.
    # Numpy reshape uses C like indexing by default.
    m = np.reshape(m, (t, j * ax))

    print(m.shape)
    print(m)

    # for rep_i in range(args.num_repetitions):
    # save_file = sample_file_template.format(sample_i)
    # animation_save_path = os.path.join(out_path, str(sample_i), save_file)
    from_array(arr=m, sampling_frequency=fps)  # , file_name=animation_save_path)



