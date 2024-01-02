#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from visualize.vicon_visualization import from_array
import utils.karate.data_info as data_info
from scipy import linalg
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_modiffae_and_diffusion, load_model, calculate_embeddings
from utils.model_util import create_semantic_regressor
from load.get_data import get_dataset_loader
from utils.parser_util import manipulation_quantitative_evaluation_args
from utils.parser_util import model_parser, get_model_path_from_args
import torch
from evaluation.fid import calculate_fid
from sample.rejection_generation import generate_samples_for_data_loader, create_attribute_labels
import copy
import matplotlib.pyplot as plt
from evaluation.guided_manipulation import manipulate_single_sample

grade_number_to_name = {
    0: '9 kyu',
    1: '8 kyu',
    2: '7 kyu',
    3: '6 kyu',
    4: '5 kyu',
    5: '4 kyu',
    6: '3 kyu',
    7: '2 kyu',
    8: '1 kyu',
    9: '1 dan',
    10: '2 dan',
    11: '3 dan',
    12: '4 dan'
}

technique_class_to_name = {
    0: 'Reverse punch',   # reverse punch
    1: 'Front kick',   # front kick
    2: 'Low roundhouse kick',   # roundhouse kick at knee to hip height
    3: 'High roundhouse kick',   # roundhouse kick at shoulder to (top) head height
    4: 'Spinning back kick'   # spinning back kick
}


def load_modiffae(modiffae_model_path):
    modiffae_args = model_parser(model_type="modiffae", model_path=modiffae_model_path)

    modiffae_validation_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='validation'
    )

    modiffae_model, modiffae_diffusion = create_modiffae_and_diffusion(modiffae_args, modiffae_validation_data)

    print(f"Loading checkpoints from [{modiffae_model_path}]...")
    modiffae_state_dict = torch.load(modiffae_model_path, map_location='cpu')
    load_model(modiffae_model, modiffae_state_dict)

    modiffae_model.to(dist_util.dev())
    modiffae_model.eval()

    return modiffae_model, modiffae_diffusion, modiffae_args


def load_semantic_regressor_ckpt(semantic_regressor_model_path, semantic_encoder):
    semantic_regressor_args = model_parser(model_type="semantic_regressor", model_path=semantic_regressor_model_path)

    # Important to use train data here! Regressor calculates z parameters based on them.
    semantic_regressor_train_data = get_dataset_loader(
        name=semantic_regressor_args.dataset,
        batch_size=semantic_regressor_args.batch_size,
        num_frames=semantic_regressor_args.num_frames,
        test_participant=semantic_regressor_args.test_participant,
        pose_rep=semantic_regressor_args.pose_rep,
        split='train'
    )

    semantic_regressor_model = create_semantic_regressor(
        semantic_regressor_args,
        semantic_regressor_train_data,
        semantic_encoder
    )

    print(f"Loading checkpoints from [{semantic_regressor_model_path}]...")
    semantic_regressor_state_dict = torch.load(semantic_regressor_model_path, map_location='cpu')
    load_model(semantic_regressor_model, semantic_regressor_state_dict)

    semantic_regressor_model.to(dist_util.dev())
    semantic_regressor_model.eval()

    return semantic_regressor_model, semantic_regressor_args

def extract_data_for_labels(data, technique_cls, grade_name):
    data_loader_copy = copy.deepcopy(data)
    data_loader_copy.dataset.reduce_to_data_for_labels(technique_cls, grade_name)
    return data_loader_copy


def calc_checkpoint_fid(semantic_generator_model_path, modiffae_model, modiffae_diffusion):
    semantic_generator_model, semantic_generator_diffusion, semantic_generator_args = (
        load_semantic_generator_ckpt(semantic_generator_model_path))

    semantic_generator_train_data = get_dataset_loader(
        name=semantic_generator_args.dataset,
        batch_size=semantic_generator_args.batch_size,
        num_frames=semantic_generator_args.num_frames,
        test_participant=semantic_generator_args.test_participant,
        pose_rep=semantic_generator_args.pose_rep,
        split='train'
    )

    validation_data = get_dataset_loader(
        name=semantic_generator_args.dataset,
        batch_size=semantic_generator_args.batch_size,
        num_frames=semantic_generator_args.num_frames,
        test_participant=semantic_generator_args.test_participant,
        pose_rep=semantic_generator_args.pose_rep,
        split='validation'
    )

    grade_nr_to_label = lambda grade: (1 / (13 - 1)) * grade

    nr_of_combinations = 5 * 13
    count = 0
    ckpt_combination_fids = []

    for technique_cls in range(5):
        for grade_nr, grade_name in grade_number_to_name.items():
            combination_data_loader = extract_data_for_labels(validation_data, technique_cls, grade_name)

            generated_samples = generate_samples_for_data_loader(
                models=((modiffae_model, modiffae_diffusion),
                        (semantic_generator_model, semantic_generator_diffusion)),
                n_frames=semantic_generator_args.num_frames,
                batch_size=semantic_generator_args.batch_size,
                one_hot_labels=create_attribute_labels(
                    grade_nr_to_label(grade_nr), technique_cls, semantic_generator_args.batch_size),
                modiffae_latent_dim=semantic_generator_args.modiffae_latent_dim,
                data=semantic_generator_train_data.dataset
            )

            generated_samples_data_loader = get_dataset_loader(
                name=semantic_generator_args.dataset,
                batch_size=semantic_generator_args.batch_size,
                num_frames=semantic_generator_args.num_frames,
                test_participant=semantic_generator_args.test_participant,
                pose_rep=semantic_generator_args.pose_rep,
                split='train',
                data_array=generated_samples
            )

            combination_semantic_embeddings = calculate_embeddings(combination_data_loader,
                                                                   modiffae_model.semantic_encoder)
            generated_samples_semantic_embeddings = calculate_embeddings(generated_samples_data_loader,
                                                                         modiffae_model.semantic_encoder)
            combination_fid = calculate_fid(combination_semantic_embeddings, generated_samples_semantic_embeddings)
            ckpt_combination_fids.append(combination_fid)
            count += 1
            print(f'Progress for ckpt: {count}/{nr_of_combinations}')

    ckpt_mean_fid = np.mean(ckpt_combination_fids)
    return ckpt_mean_fid


def calc_fid_combinations(train_data, validation_data, original_grade_name, original_technique_cls,
                          semantic_encoder, test_embeddings):
    technique_combination_fid_list = []
    for tech_nr, tech_name in technique_class_to_name.items():
        combination_comparison_data = extract_data_for_labels(validation_data, tech_nr, original_grade_name)
        combination_comparison_data_embeddings = calculate_embeddings(combination_comparison_data,
                                                                      semantic_encoder)

        #print(combination_comparison_data_embeddings.shape)

        '''combination_comparison_data_train = extract_data_for_labels(train_data, tech_nr, original_grade_name)
        combination_comparison_data_embeddings_train = calculate_embeddings(combination_comparison_data_train,
                                                                            semantic_encoder)

        intra_fid = calculate_fid(combination_comparison_data_embeddings_train,
                                  combination_comparison_data_embeddings)
        if tech_nr == original_technique_cls:
            print(intra_fid)'''

        #print(combination_comparison_data_embeddings_train.shape)

        '''combination_comparison_data_embeddings = torch.cat((combination_comparison_data_embeddings,
                                                            combination_comparison_data_embeddings_train))'''

        #print(combination_comparison_data_embeddings.shape)

        #exit()

        technique_combination_fid = calculate_fid(test_embeddings,
                                                  combination_comparison_data_embeddings)
        technique_combination_fid_list.append(technique_combination_fid)

    #print(technique_combination_fid_list)

    # Calculate distances to all the grades of the current technique (most likely grade should
    # remain the same when manipulating technique).
    grade_combination_fid_list = []
    for grade_nr, grade_name in grade_number_to_name.items():
        combination_comparison_data = extract_data_for_labels(validation_data, original_technique_cls, grade_name)
        combination_comparison_data_embeddings = calculate_embeddings(combination_comparison_data,
                                                                      semantic_encoder)

        '''combination_comparison_data_train = extract_data_for_labels(train_data, original_technique_cls, grade_name)
        combination_comparison_data_embeddings_train = calculate_embeddings(combination_comparison_data_train,
                                                                            semantic_encoder)

        intra_fid = calculate_fid(combination_comparison_data_embeddings_train,
                                  combination_comparison_data_embeddings)

        if grade_name == original_grade_name:
            print(intra_fid)'''

        '''combination_comparison_data_embeddings = torch.cat((combination_comparison_data_embeddings,
                                                            combination_comparison_data_embeddings_train))'''

        grade_combination_fid = calculate_fid(test_embeddings,
                                              combination_comparison_data_embeddings)
        grade_combination_fid_list.append(grade_combination_fid)

    #print(grade_combination_fid_list)

    return technique_combination_fid_list, grade_combination_fid_list


def main():
    args = manipulation_quantitative_evaluation_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    modiffae_model_path = args.modiffae_model_path
    modiffae_model, modiffae_diffusion, modiffae_args = load_modiffae(modiffae_model_path)

    semantic_regressor_model_path = args.semantic_regressor_model_path
    semantic_regressor_model, _ = (
        load_semantic_regressor_ckpt(semantic_regressor_model_path, modiffae_model.semantic_encoder))

    test_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='test'
    )

    # I think it would be better to combine both train and validation
    validation_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='validation'
    )

    train_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='train'
    )

    # Specify which case to evaluate

    print(modiffae_args.test_participant)

    if modiffae_args.test_participant == 'b0401':
        original_grade_nr = 1
        t_g = 9
        neg = False
    elif modiffae_args.test_participant == 'b0372':
        original_grade_nr = 9
        t_g = 1
        neg = True
    else:
        raise ValueError('Invalid test participant')

    # Technique example (bad)
    '''original_technique_cls = 1  # To filter out the according techniques from the test participant
    #original_grade = None
    target_technique_idx = 0
    target_grade_nr = original_grade_nr #(1 / 12) * original_grade_nr
    manipulated_attribute_idx = 0
    is_negative = False
    batch_size = modiffae_args.batch_size'''

    # Technique example (good)
    '''original_technique_cls = 3  # To filter out the according techniques from the test participant
    # original_grade = None
    target_technique_idx = 1
    target_grade_nr = original_grade_nr  # (1 / 12) * original_grade_nr
    manipulated_attribute_idx = 1
    is_negative = False
    batch_size = modiffae_args.batch_size'''

    print(t_g, neg)

    # Grade example (bad but justified)
    original_technique_cls = 1
    target_technique_idx = 1
    target_grade_nr = t_g
    manipulated_attribute_idx = 5
    is_negative = neg
    batch_size = modiffae_args.batch_size


    '''if manipulated_attribute_idx == 5:
        mode = 'grade'
    else:
        mode = "technique" '''

    #if mode == 'technique':
    original_grade_name = grade_number_to_name[original_grade_nr]
    # Iterate over techniques and calculate the distance to the current technique. (Most likely technique
    # should change when manipulating the technique)
    combination_test_data_og = extract_data_for_labels(test_data, original_technique_cls, original_grade_name)
    combination_test_data_og_embeddings = calculate_embeddings(combination_test_data_og,
                                                               modiffae_model.semantic_encoder)

    print(len(combination_test_data_og_embeddings))

    og_technique_combination_fid_list, og_grade_combination_fid_list = (
        calc_fid_combinations(train_data, validation_data, original_grade_name, original_technique_cls,
                              modiffae_model.semantic_encoder, combination_test_data_og_embeddings))



    print(og_technique_combination_fid_list)
    print(og_grade_combination_fid_list)

    #exit()

    '''technique_combination_fid_list = []
    for tech_nr, tech_name in technique_class_to_name.items():
        combination_comparison_data = extract_data_for_labels(validation_data, tech_nr, original_grade_name)
        combination_comparison_data_embeddings = calculate_embeddings(combination_comparison_data,
                                                                      modiffae_model.semantic_encoder)

        technique_combination_fid = calculate_fid(combination_test_data_og_embeddings,
                                                  combination_comparison_data_embeddings)
        technique_combination_fid_list.append(technique_combination_fid)

    print(technique_combination_fid_list)

    # Calculate distances to all the grades of the current technique (most likely grade should
    # remain the same when manipulating technique).
    grade_combination_fid_list = []
    for grade_nr, grade_name in grade_number_to_name.items():
        combination_comparison_data = extract_data_for_labels(validation_data, original_technique_cls, grade_name)
        combination_comparison_data_embeddings = calculate_embeddings(combination_comparison_data,
                                                                      modiffae_model.semantic_encoder)

        grade_combination_fid = calculate_fid(combination_test_data_og_embeddings,
                                              combination_comparison_data_embeddings)
        grade_combination_fid_list.append(grade_combination_fid)

    print(grade_combination_fid_list)'''

    ####
    combination_test_manipulated_embeddings_after_motion_creation = []
    for b, (motion, cond) in enumerate(combination_test_data_og):
        motion = motion.to(dist_util.dev())
        cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                     cond['y'].items()}

        cond_labels = cond['y']['labels']

        for i in range(motion.shape[0]):
            single_model_kwargs = copy.deepcopy(cond)
            single_model_kwargs['y'].update({'mask': cond['y']['mask'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'lengths': cond['y']['lengths'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'original_motion': cond['y']['original_motion'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'distance': cond['y']['distance'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'labels': cond['y']['labels'][i].unsqueeze(dim=0)})

            m = motion[i].unsqueeze(dim=0)
            # print(m.shape)

            #c = cond_labels[i]
            #current_technique_idx = torch.argmax(c[:5]).cpu().detach().numpy()
            # print(current_technique_idx)
            #current_grade = c[5].cpu().detach().numpy()
            #current_grade_nr = round(current_grade * 12)
            # c = c.unsqueeze(dim=0)

            # print(current_technique_idx)
            # exit()

            '''if current_grade_nr == 9:
                target_grade_nr = 1
            elif current_grade_nr == 1:
                target_grade_nr = 9
            else:
                raise ValueError(f'Unexpected garde number: {current_grade_nr}')'''

            target_grade = (1 / 12) * target_grade_nr
            # print(target_grade)

            #lower_grade = target_grade_nr == 1
            og_xyz_motion, manipulated_xyz_motion, manipulated_sample = manipulate_single_sample(
                m, single_model_kwargs, modiffae_model, modiffae_diffusion, semantic_regressor_model,
                target_technique_idx=target_technique_idx, target_grade=target_grade,
                manipulated_attribute_idx=manipulated_attribute_idx,
                is_negative=is_negative, batch_size=batch_size, return_rot_sample=True
            )

            print(manipulated_xyz_motion.shape)
            print(manipulated_sample.shape)

            with torch.no_grad():
                manipulated_emb = modiffae_model.semantic_encoder(manipulated_sample)
            combination_test_manipulated_embeddings_after_motion_creation.append(manipulated_emb)

            '''from_array(arr=og_xyz_motion, mode='inspection')
            from_array(arr=manipulated_xyz_motion, mode='inspection')'''
    combination_test_manipulated_embeddings_after_motion_creation = (
        torch.cat(combination_test_manipulated_embeddings_after_motion_creation, dim=0))
            #exit()
    ####

    print(combination_test_manipulated_embeddings_after_motion_creation.shape)

    target_grade_name = grade_number_to_name[target_grade_nr]
    target_technique_cls = target_technique_idx

    manipulated_technique_combination_fid_list, manipulated_grade_combination_fid_list = (
        calc_fid_combinations(train_data, validation_data, target_grade_name, target_technique_cls,
                              modiffae_model.semantic_encoder,
                              combination_test_manipulated_embeddings_after_motion_creation))

    print(manipulated_technique_combination_fid_list)
    print(manipulated_grade_combination_fid_list)

    exit()


        # - manipulate test data for only from the current technique into the desired
        # - repeat step 1 (the first two loops). But calc distance between grades and the desired technique


    '''else:
        # same but for grade instead
        pass'''


    exit()


    '''_, model_name = os.path.split(modiffae_model_path)
    model_name = model_name.split('.')[0]
    base_dir, _ = os.path.split(os.path.dirname(modiffae_model_path))
    test_participant = modiffae_args.test_participant

    semantic_generator_dir = (
        os.path.join(base_dir, f'semantic_generator_based_on_modiffae_{test_participant}_{model_name}'))

    checkpoints = [p for p in sorted(os.listdir(semantic_generator_dir)) if p.startswith('model') and p.endswith('.pt')]'''

    #ckpt_average_fids = []
    #for ckpt in checkpoints:
    #ckpt_path = os.path.join(semantic_generator_dir, ckpt)
    ckpt_average_fid = calc_checkpoint_fid(
        semantic_generator_model_path=ckpt_path,
        modiffae_model=modiffae_model,
        modiffae_diffusion=modiffae_diffusion
    )
    #    ckpt_average_fids.append(ckpt_average_fid)

    eval_dir = os.path.join(semantic_generator_dir, "evaluation")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    npy_file_path = os.path.join(eval_dir, "average_fids")
    np.save(npy_file_path, np.array(ckpt_average_fids))

    '''f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(8)

    checkpoints_for_plt = [str(int(int(ch.strip("model").strip(".pt")) / 1000)) + "k" for ch in checkpoints]
    plt.plot(checkpoints_for_plt, ckpt_average_fids)

    plt_file_path = os.path.join(eval_dir, "average_fids")
    plt.savefig(plt_file_path)'''


if __name__ == "__main__":
    main()
