import numpy as np
import os
from load.dataset import Dataset

import torch

class KaratePoses(Dataset):
    def __init__(self, data_path="datasets/karate", split="train", **kwargs):
        # TODO: adjust max number of frames parameter after preprocessing (currently 125) -> set to 100 meaning 4 sec
        super().__init__(**kwargs)

        self.data_name = "karate"
        data_file_path = os.path.join(data_path, "karate_motion_unmodified.npy")
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
