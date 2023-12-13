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

    #if mode == 'grade':
    technique_distance = 1 - candidate_technique_prediction
    grade_distance = torch.abs(target_grade - candidate_grade_prediction)
    candidate_score = (technique_distance + grade_distance) / 2
    #else:
    #    technique_distance = 1 - candidate_technique_prediction
        #grade_distance = torch.abs(target_grade - candidate_grade_prediction)
        #candidate_score = (technique_distance + grade_distance) / 2
    #    candidate_score = technique_distance

    #print(candidate_technique_prediction, candidate_grade_prediction)


    if mode == 'grade':
        if is_negative and candidate_grade_prediction < target_grade:
            return None
        elif not is_negative and candidate_grade_prediction > target_grade:
            return None

    #print(candidate_prediction, target_technique_idx, target_grade)

    #print(technique_distance, grade_distance, candidate_score)
    #exit()
    return candidate_score

def manipulate_single_sample(sample, single_model_kwargs, modiffae_model, modiffae_diffusion, semantic_regressor_model,
                             target_technique_idx, target_grade, manipulated_attribute_idx, is_negative, batch_size):

    # TODO:
    #  - extract weight of manipulated_attribute_idx
    #  - create many manipulation with weights between 0 and lambda max (maybe smaller lambda max for
    #  grade and larger for technique? But then again, we also want to hit the right garde in the technique)
    #  Try first with the same lambda amx and see how the results are.
    #  - create the samples, predict their labels and take the one which is closest (with regard to the target
    #  technique distance and the target grade distance)
    #  - if lower_grade is true, the target grade attribute has the label 0, if false then 1

    if is_negative:
        manipulation_sign = -1
    else:
        manipulation_sign = 1

    if manipulated_attribute_idx == 5:
        mode = 'grade'
        # see if this is enough
        lambda_max = 0.008
    else:
        mode = 'technique'
        # TODO: try out and find good value
        lambda_max = 0.1

    #print(f'idx {manipulated_attribute_idx}')

    lambda_steps = [(lambda_max / (batch_size - 1)) * i for i in range(batch_size)]

    diffuse_function = modiffae_diffusion.ddim_reverse_sample_loop
    # forward pass from the stochastic encoder
    single_noise = diffuse_function(
        modiffae_model,
        sample,
        clip_denoised=False,
        model_kwargs=single_model_kwargs
    )

    #print(single_noise["sample"].shape)

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

        #candidate_technique_prediction = F.sigmoid(class_after)
        #candidate_grade_prediction =

        candidate_prediction = torch.sigmoid(candidate_prediction)
        candidate_technique_prediction = candidate_prediction[:5]
        candidate_technique_prediction = F.softmax(candidate_technique_prediction, dim=-1)
        candidate_technique_prediction = candidate_technique_prediction[target_technique_idx]
        candidate_grade_prediction = candidate_prediction[5]
        #candidate_grade_prediction = torch.sigmoid(candidate_grade_prediction)

        #print(candidate_technique_prediction, candidate_grade_prediction)

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
    ####
    #print(candidate_embeddings_with_scores[:, 1])
    #for sc in candidate_embeddings_with_scores:
    #    print(sc[0], sc[2])

    try:
        best_candidate = min(candidate_embeddings_with_scores, key=lambda x: x[0])[1]
    except ValueError:
        print("Warning: Not better candidate found than the original. This only happens "
              "if the grade is manipulated but the original motion is already beyond the goal.")
        denormalized_og_semantic_embedding = semantic_regressor_model.denormalize(normalized_semantic_embedding)
        best_candidate = denormalized_og_semantic_embedding

    #print(best_candidate.shape)

    #exit()


    #candidate_embeddings.append(candidate)

    #candidate_predictions = semantic_regressor_model.regressor(torch.as_tensor(candidate_embeddings,
    #                                                                           device=dist_util.dev()))
    #print(candidate_predictions.shape)
    #exit()

    '''class_after = semantic_regressor_model.regressor(semantic_regressor_model.normalize(best_candidate))
    class_after = F.sigmoid(class_after)
    class_after_technique = F.softmax(class_after[:, :5], dim=-1)
    print(class_after_technique)
    # class_after_skill_level = F.softmax(class_after[:, 5:])
    class_after_skill_level = class_after[:, 5]
    print(class_after_skill_level)'''

    #exit()

    #########

    single_model_kwargs['y']['semantic_emb'] = best_candidate

    sample_fn = modiffae_diffusion.ddim_sample_loop

    # Using the diffused data from the encoder in the form of noise
    manipulated_single_sample = sample_fn(
        modiffae_model,
        # (args.batch_size, model.num_joints, model.num_feats, n_frames),
        #(args.batch_size, data.dataset.num_joints, data.dataset.num_feats, n_frames),
        (1, 39, 6, 100),
        clip_denoised=False,
        model_kwargs=single_model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        # 'sample' should be xT.
        noise=single_noise['sample'],
        # noise=None,
        const_noise=False,
    )

    #print(manipulated_single_sample.shape)

    rot2xyz_mask = single_model_kwargs['y']['mask'].reshape(1, 100).bool()

    og_xyz = modiffae_model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=modiffae_model.pose_rep, glob=True,
                                    translation=True, jointstype='karate', vertstrans=True, betas=None,
                                    beta=0, glob_rot=None, get_rotations_back=False,
                                    distance=single_model_kwargs['y']['distance'])
    og_xyz = og_xyz.cpu().detach().numpy()[0]
    og_xyz = og_xyz.transpose(2, 0, 1)
    t, j, ax = og_xyz.shape
    og_xyz_motion = np.reshape(og_xyz, (t, j * ax))
    #from_array(arr=og_xyz_motion, mode='inspection')

    manipulated_xyz = modiffae_model.rot2xyz(x=manipulated_single_sample, mask=rot2xyz_mask, pose_rep=modiffae_model.pose_rep, glob=True,
                                        translation=True, jointstype='karate', vertstrans=True, betas=None,
                                        beta=0, glob_rot=None, get_rotations_back=False,
                                        distance=single_model_kwargs['y']['distance'])
    manipulated_xyz = manipulated_xyz.cpu().detach().numpy()[0]
    manipulated_xyz = manipulated_xyz.transpose(2, 0, 1)
    t, j, ax = manipulated_xyz.shape
    manipulated_xyz_motion = np.reshape(manipulated_xyz, (t, j * ax))
    #from_array(arr=manipulated_xyz_motion, mode='inspection')

    return og_xyz_motion, manipulated_xyz_motion
