import random

import numpy as np
import torch
# from utils.action_label_to_idx import action_label_to_idx
#from data_loaders.tensors import collate
from load.tensors import collate
from utils.misc import to_torch
import utils.rotation_conversions as geometry


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=1, sampling="conseq", sampling_step=1, split="train",
                 pose_rep="rot_6d", num_joints=39, translation=True, root_joint_idx=5,
                 max_len=-1, min_len=-1, num_seq_max=-1, **kwargs):
        self.num_frames = num_frames
        self.sampling = sampling
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.num_joints = num_joints
        self.translation = translation
        self.max_len = max_len
        self.min_len = min_len
        self.num_seq_max = num_seq_max
        self.root_joint_idx = root_joint_idx

        if pose_rep == "rot_vec":
            self.num_feats = 3
        elif pose_rep == "rot_mat":
            self.num_feats = 9
        elif pose_rep == "rot_quat":
            self.num_feats = 4
        elif pose_rep == "rot_6d":
            self.num_feats = 6
        else:
            raise NotImplementedError("This pose representation is not implemented.")

        if self.split not in ["train", "validation", "test"]:
            raise ValueError(f"{self.split} is not a valid split")

        super().__init__()

        # to remove shuffling
        self._original_train = None
        self._original_test = None

    def action_to_label(self, action):
        return self._action_to_label[action]

    def label_to_action(self, label):
        import numbers
        if isinstance(label, numbers.Integral):
            return self._label_to_action[label]
        else:  # if it is one hot vector
            label = np.argmax(label)
            return self._label_to_action[label]

    def get_pose_data(self, data_index, frame_ix):
        pose = self._load(data_index, frame_ix)
        action = self.get_label(data_index)
        # Added for karate
        distances = None
        if getattr(self, "data_name", None) is not None and self.data_name == "karate": 
            distances = self._joint_distances[data_index]
            distances = torch.as_tensor(distances)

        labels = None
        if getattr(self, "_load_labels", None) is not None:
            labels = self._load_labels(data_index)
        return pose, action, distances, labels

    def get_label(self, ind):
        action = self.get_action(ind)
        return self.action_to_label(action)

    def get_action(self, ind):
        return self._actions[ind]

    def action_to_action_name(self, action):
        return self._action_classes[action]

    def action_name_to_action(self, action_name):
        # self._action_classes is either a list or a dictionary. If it's a dictionary, we 1st convert it to a list
        all_action_names = self._action_classes
        if isinstance(all_action_names, dict):
            all_action_names = list(all_action_names.values())
            # the keys should be ordered from 0 to num_actions
            assert list(self._action_classes.keys()) == list(range(len(all_action_names)))

        sorter = np.argsort(all_action_names)
        actions = sorter[np.searchsorted(all_action_names, action_name, sorter=sorter)]
        return actions

    def __getitem__(self, index):
        if self.split == 'train':
            data_index = self._train[index]
        else:
            data_index = self._test[index]
        return self._get_item_data_index(data_index)

    def _load(self, ind, frame_ix):
        ret = None
        ret_tr = None
        pose_rep = self.pose_rep
        if pose_rep == "xyz" or self.translation:
            if getattr(self, "_load_joints", None) is not None:
                joints_3d = self._load_joints(ind, frame_ix)
                ret = to_torch(joints_3d)
                if self.translation:
                    # Locate the root joint of all frames.
                    ret_tr = ret[:, self.root_joint_idx, :]
            else:
                raise ValueError("Joint positions could not be loaded.")

        if pose_rep != "xyz":
            if getattr(self, "_load_rot_vec", None) is None:
                raise ValueError("This representation is not possible.")
            else:
                pose = self._load_rot_vec(ind, frame_ix)
                pose = to_torch(pose)

                if pose_rep == "rot_vec":
                    ret = pose
                elif pose_rep == "rot_mat":
                    ret = geometry.axis_angle_to_matrix(pose).view(*pose.shape[:2], 9)
                elif pose_rep == "rot_quat":
                    ret = geometry.axis_angle_to_quaternion(pose)
                elif pose_rep == "rot_6d":
                    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))

        if pose_rep != "xyz" and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr
            # Putting the position at the end so that the indices of the joints stay the same.
            ret = torch.cat((ret, padded_tr[:, None]), 1)
        ret = ret.permute(1, 2, 0).contiguous()

        return ret.float()

    def _get_item_data_index(self, data_index):
        nframes = self._num_frames_in_video[data_index]

        #print(self.num_frames)

        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            frame_ix = np.arange(nframes)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(nframes, self.max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len

            if num_frames > nframes:
                '''fair = False  # True
                if fair:
                    # distills redundancy everywhere
                    choices = np.random.choice(range(nframes),
                                               num_frames,
                                               replace=True)
                    frame_ix = sorted(choices)'''
                #else:
                # adding the last frame until done
                ntoadd = max(0, num_frames - nframes)
                lastframe = nframes - 1
                padding = lastframe * np.ones(ntoadd, dtype=int)
                frame_ix = np.concatenate((np.arange(0, nframes),
                                           padding))

            '''elif self.sampling in ["conseq", "random_conseq"]:

                print('happening')

                #print(nframes, num_frames)
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(nframes),
                                           num_frames,
                                           replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")'''

        inp, action, distances, labels = self.get_pose_data(data_index, frame_ix)

        #output = {'inp': inp, 'action': action, 'dist': distances, 'labels': labels}
        output = {'inp': inp, 'dist': distances, 'labels': labels}

        if hasattr(self, '_actions') and hasattr(self, '_action_classes'):
            output['action_text'] = self.action_to_action_name(self.get_action(data_index))

        return output

    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            return min(len(self._train), num_seq_max)
        else:
            return min(len(self._test), num_seq_max)

    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test
