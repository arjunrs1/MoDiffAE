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
from utils.model_util import create_semantic_generator_and_diffusion
from load.get_data import get_dataset_loader
from utils.parser_util import generation_args, evaluation_args, generation_evaluation_args
from utils.parser_util import model_parser, get_model_path_from_args
import torch
from evaluation.fid import calculate_fid
from sample.generate import generate_samples_for_data_loader, create_attribute_labels
import copy
import matplotlib.pyplot as plt

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


def load_semantic_generator_ckpt(semantic_generator_model_path):
    semantic_generator_args = model_parser(model_type="semantic_generator", model_path=semantic_generator_model_path)

    semantic_generator_model, semantic_generator_diffusion = (
        create_semantic_generator_and_diffusion(semantic_generator_args))

    print(f"Loading checkpoints from [{semantic_generator_model_path}]...")
    semantic_generator_state_dict = torch.load(semantic_generator_model_path, map_location='cpu')
    load_model(semantic_generator_model, semantic_generator_state_dict)

    semantic_generator_model.to(dist_util.dev())
    semantic_generator_model.eval()

    return semantic_generator_model, semantic_generator_diffusion, semantic_generator_args


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


def main():
    args = generation_evaluation_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    modiffae_model_path = args.modiffae_model_path
    modiffae_model, modiffae_diffusion, modiffae_args = load_modiffae(modiffae_model_path)

    _, model_name = os.path.split(modiffae_model_path)
    model_name = model_name.split('.')[0]
    base_dir, _ = os.path.split(os.path.dirname(modiffae_model_path))
    test_participant = modiffae_args.test_participant

    semantic_generator_dir = (
        os.path.join(base_dir, f'semantic_generator_based_on_modiffae_{test_participant}_{model_name}'))

    checkpoints = [p for p in sorted(os.listdir(semantic_generator_dir)) if p.startswith('model') and p.endswith('.pt')]

    ckpt_average_fids = []
    for ckpt in checkpoints:
        ckpt_path = os.path.join(semantic_generator_dir, ckpt)
        ckpt_average_fid = calc_checkpoint_fid(
            semantic_generator_model_path=ckpt_path,
            modiffae_model=modiffae_model,
            modiffae_diffusion=modiffae_diffusion
        )
        ckpt_average_fids.append(ckpt_average_fid)

    eval_dir = os.path.join(semantic_generator_dir, "evaluation")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    npy_file_path = os.path.join(eval_dir, "average_fids")
    np.save(npy_file_path, np.array(ckpt_average_fids))

    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(8)

    checkpoints_for_plt = [str(int(int(ch.strip("model").strip(".pt")) / 1000)) + "k" for ch in checkpoints]
    plt.plot(checkpoints_for_plt, ckpt_average_fids)

    plt_file_path = os.path.join(eval_dir, "average_fids")
    plt.savefig(plt_file_path)


if __name__ == "__main__":
    main()
