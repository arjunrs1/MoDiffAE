import copy

from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import model_parser, manipulation_qualitative_evaluation_args
from utils.model_util import create_modiffae_and_diffusion, load_model, calculate_embeddings
from utils.model_util import create_semantic_regressor
from utils import dist_util
from load.get_data import get_dataset_loader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
import seaborn as sns
import colorcet as cc
import pandas as pd
import json
import torch.nn.functional as F
from evaluation.guided_manipulation import manipulate_single_sample

from evaluation.guided_manipulation import load_modiffae, load_semantic_regressor_ckpt


technique_class_to_name = {
    0: 'reverse_punch',   # reverse punch
    1: 'front_kick',   # front kick
    2: 'low_roundhouse_kick',   # roundhouse kick at knee to hip height
    3: 'high_roundhouse_kick',   # roundhouse kick at shoulder to (top) head height
    4: 'spinning_back_kick'   # spinning back kick
}


def manipulate():
    pass


def create_directories(test_participant):
    cwd = os.getcwd()
    base_dir = os.path.join(cwd, 'evaluation', 'test_manipulations', test_participant)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    grade_dir = os.path.join(base_dir, 'grade')
    if not os.path.exists(grade_dir):
        os.mkdir(grade_dir)

    for t in technique_class_to_name.keys():
        t_dir = os.path.join(grade_dir, f'{technique_class_to_name[t]}')
        if not os.path.exists(t_dir):
            os.mkdir(t_dir)

    technique_dir = os.path.join(base_dir, 'technique')
    if not os.path.exists(technique_dir):
        os.mkdir(technique_dir)

    for t in technique_class_to_name.keys():
        t_dir = os.path.join(technique_dir, f'from_{technique_class_to_name[t]}')
        if not os.path.exists(t_dir):
            os.mkdir(t_dir)

        for r in technique_class_to_name.keys():
            if r != t:
                r_dir = os.path.join(t_dir, f'to_{technique_class_to_name[r]}')
                if not os.path.exists(r_dir):
                    os.mkdir(r_dir)


def main():

    args = manipulation_qualitative_evaluation_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    modiffae_model_path = args.modiffae_model_path
    modiffae_model, modiffae_diffusion, modiffae_args = load_modiffae(modiffae_model_path)

    create_directories(modiffae_args.test_participant)

    semantic_regressor_model, _ = (
        load_semantic_regressor_ckpt(args.semantic_regressor_model_path, modiffae_model.semantic_encoder))

    test_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='test'
    )

    cwd = os.getcwd()
    base_dir = os.path.join(cwd, 'evaluation', 'test_manipulations', modiffae_args.test_participant)

    # Grade manipulations
    for b, (motion, cond) in enumerate(test_data):
        motion = motion.to(dist_util.dev())
        cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                     cond['y'].items()}

        cond_labels = cond['y']['labels']
        '''print(cond['y']['mask'].shape)
        print(cond['y']['lengths'].shape)
        print(cond['y']['original_motion'].shape)
        print(cond['y']['distance'].shape)
        print(cond['y']['labels'].shape)

        print(cond['y'].keys())'''

        for i in range(motion.shape[0]):

            single_model_kwargs = copy.deepcopy(cond)
            single_model_kwargs['y'].update({'mask': cond['y']['mask'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'lengths': cond['y']['lengths'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'original_motion': cond['y']['original_motion'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'distance': cond['y']['distance'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'labels': cond['y']['labels'][i].unsqueeze(dim=0)})

            '''print(single_model_kwargs['y']['mask'].shape)
            print(single_model_kwargs['y']['lengths'].shape)
            print(single_model_kwargs['y']['original_motion'].shape)
            print(single_model_kwargs['y']['distance'].shape)
            print(single_model_kwargs['y']['labels'].shape)'''

            m = motion[i].unsqueeze(dim=0)
            #print(m.shape)

            c = cond_labels[i]
            current_technique_idx = torch.argmax(c[:5]).cpu().detach().numpy()
            #print(current_technique_idx)
            current_grade = c[5].cpu().detach().numpy()
            current_grade_nr = round(current_grade * 12)
            #c = c.unsqueeze(dim=0)

            #print(current_technique_idx)
            #exit()

            samples_save_dir = os.path.join(base_dir, 'grade',
                                            f'{technique_class_to_name[current_technique_idx.item()]}')
            og_path = os.path.join(samples_save_dir, f'og_xyz_motion_{(i + 1) * (b + 1)}.npy')
            manipulated_path = os.path.join(samples_save_dir, f'manipulated_xyz_motion_{(i + 1) * (b + 1)}.npy')
            if not os.path.isfile(manipulated_path):

                if current_grade_nr == 9:
                    target_grade_nr = 1
                elif current_grade_nr == 1:
                    target_grade_nr = 9
                else:
                    raise ValueError(f'Unexpected garde number: {current_grade_nr}')

                target_grade = (1 / 12) * target_grade_nr
                #print(target_grade)

                lower_grade = target_grade_nr == 1
                og_xyz_motion, manipulated_xyz_motion = manipulate_single_sample(
                    m, single_model_kwargs, modiffae_model, modiffae_diffusion, semantic_regressor_model,
                    target_technique_idx=current_technique_idx, target_grade=target_grade,
                    manipulated_attribute_idx=5, is_negative=lower_grade, batch_size=args.batch_size
                )

                np.save(og_path, og_xyz_motion)
                np.save(manipulated_path, manipulated_xyz_motion)
                print(f'Saved files in {samples_save_dir}')

    ###################

    # Technique manipulations
    for b, (motion, cond) in enumerate(test_data):
        motion = motion.to(dist_util.dev())
        cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                     cond['y'].items()}

        cond_labels = cond['y']['labels']
        '''print(cond['y']['mask'].shape)
        print(cond['y']['lengths'].shape)
        print(cond['y']['original_motion'].shape)
        print(cond['y']['distance'].shape)
        print(cond['y']['labels'].shape)

        print(cond['y'].keys())'''

        for i in range(motion.shape[0]):

            single_model_kwargs = copy.deepcopy(cond)
            single_model_kwargs['y'].update({'mask': cond['y']['mask'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'lengths': cond['y']['lengths'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update(
                {'original_motion': cond['y']['original_motion'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'distance': cond['y']['distance'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'labels': cond['y']['labels'][i].unsqueeze(dim=0)})

            '''print(single_model_kwargs['y']['mask'].shape)
            print(single_model_kwargs['y']['lengths'].shape)
            print(single_model_kwargs['y']['original_motion'].shape)
            print(single_model_kwargs['y']['distance'].shape)
            print(single_model_kwargs['y']['labels'].shape)'''

            m = motion[i].unsqueeze(dim=0)
            # print(m.shape)

            c = cond_labels[i]
            current_technique_idx = torch.argmax(c[:5]).cpu().detach().numpy()
            # print(current_technique_idx)
            current_grade = c[5].cpu().detach().numpy()
            #current_grade_nr = round(current_grade * 12)
            # c = c.unsqueeze(dim=0)

            # print(current_technique_idx)
            # exit()

            for t in technique_class_to_name.keys():
                if t != current_technique_idx:

                    samples_save_dir = os.path.join(base_dir, 'technique',
                                                    f'from_{technique_class_to_name[current_technique_idx.item()]}',
                                                    f'to_{technique_class_to_name[t]}')
                    og_path = os.path.join(samples_save_dir, f'og_xyz_motion_{(i + 1) * (b + 1)}.npy')
                    manipulated_path = os.path.join(samples_save_dir,
                                                    f'manipulated_xyz_motion_{(i + 1) * (b + 1)}.npy')
                    if not os.path.isfile(manipulated_path):

                        '''if current_grade_nr == 9:
                            target_grade_nr = 1
                        elif current_grade_nr == 1:
                            target_grade_nr = 9
                        else:
                            raise ValueError(f'Unexpected garde number: {current_grade_nr}')'''

                        #target_grade = (1 / 12) * target_grade_nr

                        # print(target_grade)

                        print(current_technique_idx, t)

                        #lower_grade = target_grade_nr == 1
                        og_xyz_motion, manipulated_xyz_motion = manipulate_single_sample(
                            m, single_model_kwargs, modiffae_model, modiffae_diffusion, semantic_regressor_model,
                            target_technique_idx=t, target_grade=current_grade.item(),
                            manipulated_attribute_idx=t, is_negative=False, batch_size=args.batch_size
                        )

                        np.save(og_path, og_xyz_motion)
                        np.save(manipulated_path, manipulated_xyz_motion)
                        print(f'Saved files in {samples_save_dir}')




            #exit()

            # TODO: manipulate for each of the target techniques
            #target_technique_idxs = [i for i in range(5) if i != current_technique_idx]
            #print(target_technique_idxs)

            #print(current_grade, current_grade_nr)






            #print(c)



            #exit()



        #with torch.no_grad():
        #    og_motion = cond['y']['original_motion']


if __name__ == "__main__":
    main()
