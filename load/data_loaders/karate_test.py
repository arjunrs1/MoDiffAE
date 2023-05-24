import numpy as np
import os
from load.dataset import Dataset

from load.data_loaders.karate import KaratePoses
from utils.parser_util import classify_args
from utils.fixseed import fixseed
from load.get_data import get_dataset_loader

import torch
'''
class KaratePoses(Dataset):
    def __init__(self, data_path="datasets/karate", split="train", **kwargs):
        # TODO: adjust max number of frames parameter after preprocessing (currently 125) -> set to 100 meaning 4 sec
        super().__init__(**kwargs)

        self.data_name = "karate"
        data_file_path = os.path.join(data_path, "karate_motion_modified.npy")
        data = np.load(data_file_path, allow_pickle=True)

        self._pose = [x for x in data["joint_axis_angles"]]
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

        print(data[0])
        exit()

    def _load_joints(self, ind, frame_ix):
        return self._joints[ind][frame_ix].reshape(-1, 39, 3)

    def _load_rot_vec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 38, 3)
        return pose


karate_action_enumerator = {
    0: 'Gyaku-Zuki',
    1: 'Mae-Geri',
    2: 'Mawashi-Geri gedan',
    3: 'Mawashi-Geri jodan',
    4: 'Ushiro-Mawashi-Geri'
}

'''


def load_dataset(args, max_frames, n_frames):
    ds_loader = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test')
    ds_loader.fixed_length = n_frames
    return ds_loader


if __name__ == "__main__":

    kp = KaratePoses()
    kp.num_frames = 100
    # print(kp._pose[1].shape)
    #t = kp._load_rotvec(0, 0)
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

    ''' # this works
    samples = kp._data
    for i in range(samples['joint_positions'].shape[0]):
        print('Showing exemplary motion...')
        from_array(samples['joint_positions'][i])

        axis_angles = samples['joint_axis_angles'][i]
        start_index = data_info.joint_to_index['T10']
        start = samples['joint_positions'][i][:, start_index * 3:start_index * 3 + 3]
        distances = samples['joint_distances'][i]
        recon_pos = karate_geometry.calc_positions(
            chain_start_positions=start,
            start_label='T10',  # "LFHD",
            axis_angles=axis_angles,
            distances=distances
        )
        print('Showing the same motion but reconstructed (should be the same)...')
        from_array(recon_pos)
    '''

    pos = kp[0]
    #print(pos)

    #exit()

    #print('hi')

    #d = kp._data['joint_positions'][0]
    #print(d.shape)




    #from_array(d)

    # d_post = kp[0]
    # print(d_post['inp'].shape)

    ################
    args = classify_args()
    #args.dataset = 'karate_test'
    #print(args.dataset)

    #exit()


    fixseed(args.seed)
    max_frames = 100
    fps = 25
    n_frames = max_frames   # min(max_frames, int(args.motion_length * fps))

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)

    print(len(data.dataset))

    iterator = iter(data)

    data_batch, model_kwargs = next(iterator)

    #total_num_samples = args.num_samples * args.num_repetitions

    print(data_batch[0].shape, model_kwargs['y']['labels'][0])

    # TODO: get original_motion encode it with the semantic encoder and then train classifier with the extracted labels

    #print("Creating model and diffusion...")
    #model, diffusion = create_model_and_diffusion(args, data)

    #summary(model, [(1, 39, 6, 125), [1]])

    exit()

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
    #data_batch, model_kwargs = next(iterator)

    print('what is this')
    print(torch.div(10, 0))
    exit()

    while True:
        data_batch, model_kwargs = next(iterator)
        print(torch.any(torch.isnan(data_batch)))

    exit()

    device = torch.device("cpu")
    if torch.cuda.is_available() and dist_util.dev() != 'cpu':
        device = torch.device(dist_util.dev())
    print(f'device: {device}')
    data_batch = data_batch.to(device)
    model_kwargs['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

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

    print('starting conversion')
    distance = model_kwargs['y']['distance']

    #print(len(distance))
    #exit()

    start = time.time()
    og_xyz = model.rot2xyz(x=data_batch, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, translation=True,
                           distance=distance)
    end = time.time()
    print(f'time for reconstruction: {end - start}')
    # for a batch of 64 samples this takes around 0.0135 seconds on a rtx 3080
    # and 0.3354 seconds on the cpu

    print('ended conversion')

    og_xyz = og_xyz.cpu().numpy()

    lengths = model_kwargs['y']['lengths'].cpu().numpy()

    for i in range(10):

        length = lengths[i]
        m = og_xyz[i].transpose(2, 0, 1)[:length]

        t, j, ax = m.shape

        # It is important that the ordering is correct here.
        # Numpy reshape uses C like indexing by default.
        m = np.reshape(m, (t, j * ax))

        #print(m.shape)
        #print(m)

        # for rep_i in range(args.num_repetitions):
        # save_file = sample_file_template.format(sample_i)
        # animation_save_path = os.path.join(out_path, str(sample_i), save_file)
        from_array(arr=m, sampling_frequency=fps)  # , file_name=animation_save_path)

    print('hi')

    # x = torch.tensor([1, 1, 3]).float()
    # x[x == 2] = float('nan')

    # print(x)

    # x = geometry.add_eps_to_zero(x)
    # print(x)

    # exit()


