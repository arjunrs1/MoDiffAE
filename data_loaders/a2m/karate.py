import pickle as pkl
import numpy as np
import os
from .dataset import Dataset

import matplotlib.pyplot as plt
import scipy.stats as stats
# import seaborn as sns
import matplotlib.lines as lines


class KaratePoses(Dataset):
#class KaratePoses():
    dataname = "karate"

    def __init__(self, datapath="datasets/KaratePoses", split="train", **kargs):
        self.datapath = datapath

        # max num_frames in karate dataset is 356 (almost 15 seconds with 25 fps). 
        # But this seems too long. Check the recordings with long lengths.
        #num_frames=-2, min_len=2, max_len=356,
        # num_frames=125
        super().__init__(**kargs)

        self.data_name = "karate"

        npydatafilepath = os.path.join(datapath, "karate_motion_25_fps.npy")
        data = np.load(npydatafilepath, allow_pickle=True)

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

'''
if __name__ == "__main__":
    kp = KaratePoses()
    kp.num_frames = 125
    print(kp._pose[1].shape)
    t = kp._load_rotvec(0, 0)
    print(kp._load_joints3D(0, 0))
    print(t)

    print('max: ' + str(max(kp._num_frames_in_video)))
    print('max: ' + str(np.mean(kp._num_frames_in_video)))
    print(kp._num_frames_in_video)

    print(kp._num_frames_in_video[0])

    loaded = kp[20]
    print(loaded)
    print(loaded['inp'].shape)
    print()
'''