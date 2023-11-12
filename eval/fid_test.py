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
from load.get_data import get_dataset_loader
from utils.parser_util import generation_args, evaluation_args
from utils.parser_util import model_parser, get_model_path_from_args
import torch

def calculate_activation_statistics(activations):
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


eval_args = evaluation_args()

modiffae_args = model_parser(model_type="modiffae")

fixseed(modiffae_args.seed)
#out_path = args.output_dir

dist_util.setup_dist(modiffae_args.device)

print('Loading dataset...')
train_data = get_dataset_loader(
    name=modiffae_args.dataset,
    batch_size=modiffae_args.batch_size,
    num_frames=modiffae_args.num_frames,
    test_participant=modiffae_args.test_participant,
    pose_rep='rot_6d',#modiffae_args.pose_rep,
    split='train'
)

print("Creating model and diffusion...")
model, diffusion = create_modiffae_and_diffusion(modiffae_args, train_data)

#print(f"Loading checkpoints from [{eval_args.modiffae_model_path}]...")
#state_dict = torch.load(eval_args.modiffae_model_path, map_location='cpu')
#load_model(model, state_dict)


model.to(dist_util.dev())
# Disable random masking
model.eval()

'''cond_mean, cond_std = calculate_z_parameters(train_data, model.semantic_encoder)
semantic_regressor = SemanticRegressor(
    modiffae_latent_dim=512,
    attribute_dim=6,  # 18,
    semantic_encoder=model.semantic_encoder,
    cond_mean=cond_mean,
    cond_std=cond_std
)'''

'''regressor_path = './save/karateOnlyAir/semantic_regressor/semantic_regressor/model000050000.pt'
state_dict_reg = torch.load(regressor_path, map_location='cpu')
load_model(semantic_regressor, state_dict_reg)
# semantic_regressor.load_state_dict(state_dict_reg, strict=False)
semantic_regressor.to(dist_util.dev())'''

semantic_embeddings, labels = calculate_embeddings(train_data, model.semantic_encoder, return_labels=True)

print(semantic_embeddings.shape)

stats1 = calculate_activation_statistics(semantic_embeddings[:1500])
stats2 = calculate_activation_statistics(semantic_embeddings[1500:])
fid = calculate_fid(stats1, stats2)

print(fid)

#semantic_embeddings = semantic_regressor.normalize(semantic_embeddings)

"""data_dir = os.path.join(os.getcwd(), 'datasets', 'karate')
data_file_path = os.path.join(data_dir, "karate_motion_modified.npy")
data = np.load(data_file_path, allow_pickle=True)


data = [d for d in data if d['condition'] == 'defender']


for i, sample in enumerate(data):
    d = sample['joint_positions']
    technique_cls = sample['technique_cls']
    technique = data_info.technique_class_to_name[technique_cls]
    condition = sample['condition']
    grade = sample['grade']

    if condition == 'defender':
        print(f'Index: {i} from {len(data) - 1}')
        print(f'Technique: {technique}')
        print(f'Condition: {condition}')
        print(f'Grade: {grade}')
        from_array(d, mode='inspection')
        print('------')

# Manually add indices of problematic recordings
# found by this search. Later add these to the outlier modification
# and repeat it on the modified dataset.
found_outliers = []"""
