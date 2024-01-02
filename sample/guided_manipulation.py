from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import model_parser, regression_evaluation_args
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
import math
from visualize.vicon_visualization import from_array


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


def calc_candidate_score(candidate_technique_prediction, candidate_grade_prediction,
                         target_grade, mode, is_negative):

    technique_distance = 1 - candidate_technique_prediction
    grade_distance = torch.abs(target_grade - candidate_grade_prediction)
    candidate_score = (technique_distance + grade_distance) / 2

    if mode == 'grade':
        if is_negative and candidate_grade_prediction < target_grade:
            return None
        elif not is_negative and candidate_grade_prediction > target_grade:
            return None

    return candidate_score


def manipulate_single_sample(sample, single_model_kwargs, modiffae_model, modiffae_diffusion, semantic_regressor_model,
                             target_technique_idx, target_grade, manipulated_attribute_idx,
                             is_negative, batch_size, return_rot_sample=False):

    if is_negative:
        manipulation_sign = -1
    else:
        manipulation_sign = 1

    if manipulated_attribute_idx == 5:
        mode = 'grade'
        lambda_max = 0.008
    else:
        mode = 'technique'
        lambda_max = 0.1

    lambda_steps = [(lambda_max / (batch_size - 1)) * i for i in range(batch_size)]

    diffuse_function = modiffae_diffusion.ddim_reverse_sample_loop
    # forward pass from the stochastic encoder
    single_noise = diffuse_function(
        modiffae_model,
        sample,
        clip_denoised=False,
        model_kwargs=single_model_kwargs
    )

    original_motion = single_model_kwargs['y']['original_motion']
    with torch.no_grad():
        semantic_embedding = modiffae_model.semantic_encoder(original_motion)

    normalized_semantic_embedding = semantic_regressor_model.normalize(semantic_embedding)

    '''class_prev = semantic_regressor_model.regressor(normalized_semantic_embedding)
    class_prev = torch.sigmoid(class_prev)
    class_prev_technique = F.softmax(class_prev[:, :5], dim=-1)
    print(class_prev_technique)
    # class_prev_skill_level = F.softmax(class_prev[:, 5:])
    class_prev_skill_level = class_prev[:, 5]
    print(class_prev_skill_level)'''

    candidate_embeddings_with_scores = []
    for step in lambda_steps:
        # 0.004 was good for skill in terms of post prediction
        candidate = normalized_semantic_embedding + manipulation_sign * step * math.sqrt(512) * F.normalize(
            semantic_regressor_model.regressor.weight[manipulated_attribute_idx][None, :], dim=1)
        candidate_prediction = semantic_regressor_model.regressor(candidate).squeeze()

        candidate = semantic_regressor_model.denormalize(candidate)

        candidate_prediction = torch.sigmoid(candidate_prediction)
        candidate_technique_prediction = candidate_prediction[:5]
        candidate_technique_prediction = F.softmax(candidate_technique_prediction, dim=-1)

        candidate_technique_prediction = candidate_technique_prediction[target_technique_idx]
        candidate_grade_prediction = candidate_prediction[5]

        '''class_after = F.sigmoid(class_after)
        class_after_technique = F.softmax(class_after[:, :5])
        print(class_after_technique)
        # class_after_skill_level = F.softmax(class_after[:, 5:])
        class_after_skill_level = class_after[:, 5]'''

        candidate_score = calc_candidate_score(candidate_technique_prediction, candidate_grade_prediction,
                                               target_grade, mode, is_negative)
        if candidate_score is not None:
            candidate_embeddings_with_scores.append((candidate_score, candidate, step))

    # Choose candidate with best score
    try:
        best_candidate = min(candidate_embeddings_with_scores, key=lambda x: x[0])[1]
    except ValueError:
        print("Warning: Not better candidate found than the original. This only happens "
              "if the grade is manipulated but the original motion is already beyond the goal.")
        denormalized_og_semantic_embedding = semantic_regressor_model.denormalize(normalized_semantic_embedding)
        best_candidate = denormalized_og_semantic_embedding

    '''class_after = semantic_regressor_model.regressor(semantic_regressor_model.normalize(best_candidate))
    class_after = F.sigmoid(class_after)
    class_after_technique = F.softmax(class_after[:, :5], dim=-1)
    print(class_after_technique)
    # class_after_skill_level = F.softmax(class_after[:, 5:])
    class_after_skill_level = class_after[:, 5]
    print(class_after_skill_level)'''

    single_model_kwargs['y']['semantic_emb'] = best_candidate

    sample_fn = modiffae_diffusion.ddim_sample_loop

    # Using the diffused data from the encoder in the form of noise
    manipulated_single_sample = sample_fn(
        modiffae_model,
        (1, 39, 6, 100),
        clip_denoised=False,
        model_kwargs=single_model_kwargs,
        skip_timesteps=0,  # don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=single_noise['sample'],
        const_noise=False,
    )

    rot2xyz_mask = single_model_kwargs['y']['mask'].reshape(1, 100).bool()

    og_xyz = modiffae_model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=modiffae_model.pose_rep, glob=True,
                                    translation=True, jointstype='karate', vertstrans=True, betas=None,
                                    beta=0, glob_rot=None, get_rotations_back=False,
                                    distance=single_model_kwargs['y']['distance'])
    og_xyz = og_xyz.cpu().detach().numpy()[0]
    og_xyz = og_xyz.transpose(2, 0, 1)
    t, j, ax = og_xyz.shape
    og_xyz_motion = np.reshape(og_xyz, (t, j * ax))
    # from_array(arr=og_xyz_motion, mode='inspection')

    manipulated_xyz = modiffae_model.rot2xyz(x=manipulated_single_sample, mask=rot2xyz_mask, pose_rep=modiffae_model.pose_rep, glob=True,
                                        translation=True, jointstype='karate', vertstrans=True, betas=None,
                                        beta=0, glob_rot=None, get_rotations_back=False,
                                        distance=single_model_kwargs['y']['distance'])
    manipulated_xyz = manipulated_xyz.cpu().detach().numpy()[0]
    manipulated_xyz = manipulated_xyz.transpose(2, 0, 1)
    t, j, ax = manipulated_xyz.shape
    manipulated_xyz_motion = np.reshape(manipulated_xyz, (t, j * ax))
    # from_array(arr=manipulated_xyz_motion, mode='inspection')

    if return_rot_sample:
        return og_xyz_motion, manipulated_xyz_motion, manipulated_single_sample
    else:
        return og_xyz_motion, manipulated_xyz_motion
