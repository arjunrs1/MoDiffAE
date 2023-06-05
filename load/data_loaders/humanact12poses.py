import pickle as pkl
import numpy as np
import os
from load.dataset import Dataset


class HumanAct12Poses(Dataset):
    data_name = "humanact12"

    def __init__(self, datapath="datasets/HumanAct12Poses", split="train", pose_rep="rot_6d", num_joints=25, **kargs):
        self.datapath = datapath

        super().__init__(pose_rep=pose_rep, num_joints=num_joints, **kargs)

        self.xyz_reconstruction_mode = "smpl"

        pkldatafilepath = os.path.join(datapath, "humanact12poses.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._pose = [x for x in data["poses"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["joints3D"]]

        self._actions = [x for x in data["y"]]

        total_num_actions = 12
        self.num_actions = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanact12_coarse_action_enumerator

    def _load_joints(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rot_vec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 24, 3)
        return pose


humanact12_coarse_action_enumerator = {
    0: "warm_up",
    1: "walk",
    2: "run",
    3: "jump",
    4: "drink",
    5: "lift_dumbbell",
    6: "sit",
    7: "eat",
    8: "turn steering wheel",
    9: "phone",
    10: "boxing",
    11: "throw",
}
