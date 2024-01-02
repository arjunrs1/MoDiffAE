"""
Generate a large batch of image samples from a model and save them as a large
numpy array.
"""
import math

from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generation_args
from utils.model_util import create_modiffae_and_diffusion, load_model, create_semantic_generator_and_diffusion
from utils.model_util import create_semantic_regressor
from utils import dist_util
from load.get_data import get_dataset_loader
import shutil
from load.tensors import collate, collate_tensors
from load.data_loaders.karate import KaratePoses
from collections import Counter
import torch.nn.functional as F
from utils.karate import geometry
from utils.karate import data_info

# Karate visualization
from visualize.vicon_visualization import from_array
import random

from utils.parser_util import model_parser


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


def load_models(modiffae_model_path, semantic_generator_model_path, semantic_regressor_model_path):
    modiffae_args = model_parser(model_type="modiffae", model_path=modiffae_model_path)
    modiffae_train_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='train'
    )
    modiffae_model, modiffae_diffusion = create_modiffae_and_diffusion(modiffae_args, modiffae_train_data)
    modiffae_state_dict = torch.load(modiffae_model_path, map_location='cpu')
    load_model(modiffae_model, modiffae_state_dict)
    modiffae_model.to(dist_util.dev())
    modiffae_model.eval()

    semantic_generator_args = model_parser(model_type="semantic_generator", model_path=semantic_generator_model_path)
    semantic_generator_model, semantic_generator_diffusion = (
        create_semantic_generator_and_diffusion(semantic_generator_args))
    semantic_generator_state_dict = torch.load(semantic_generator_model_path, map_location='cpu')
    load_model(semantic_generator_model, semantic_generator_state_dict)
    semantic_generator_model.to(dist_util.dev())
    semantic_generator_model.eval()

    semantic_regressor_args = model_parser(model_type="semantic_regressor", model_path=semantic_regressor_model_path)
    semantic_regressor_train_data = get_dataset_loader(
        name=semantic_regressor_args.dataset,
        batch_size=semantic_regressor_args.batch_size,
        num_frames=semantic_regressor_args.num_frames,
        test_participant=semantic_regressor_args.test_participant,
        pose_rep=semantic_regressor_args.pose_rep,
        split='train'
    )
    semantic_encoder = modiffae_model.semantic_encoder
    semantic_regressor_model = (
        create_semantic_regressor(semantic_regressor_args, semantic_regressor_train_data, semantic_encoder))
    semantic_regressor_state_dict = torch.load(semantic_regressor_model_path, map_location='cpu')
    load_model(semantic_regressor_model, semantic_regressor_state_dict)
    semantic_regressor_model.to(dist_util.dev())
    semantic_regressor_model.eval()

    return (modiffae_model, modiffae_diffusion), \
        (semantic_generator_model, semantic_generator_diffusion), \
        semantic_regressor_model


def calc_number_of_samples_to_generate_per_grade(ratio, data):
    counts = Counter(data.get_grades())
    number_of_samples_to_generate_per_grade = {}
    for grade, count in counts.items():
        number_of_samples_to_generate_per_grade[grade] = math.ceil(count * ratio)
    return number_of_samples_to_generate_per_grade


def create_attribute_labels(grade, technique_cls, batch_size):
    skill_labels = np.array([[grade]] * batch_size)
    labels = np.array([technique_cls] * batch_size)
    one_hot_labels = np.eye(5)[labels]
    one_hot_labels = np.append(one_hot_labels, skill_labels, axis=1)
    return one_hot_labels


def generate_samples(models, n_frames, batch_size, one_hot_labels, modiffae_latent_dim, data):
    modiffae_model, modiffae_diffusion = models[0]
    semantic_generator_model, semantic_generator_diffusion = models[1]

    generator_sample_fn = semantic_generator_diffusion.ddim_sample_loop

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * batch_size
    collate_args = [dict(arg, labels=l) for
                    arg, l in zip(collate_args, one_hot_labels)]
    _, model_kwargs = collate(collate_args)
    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
                         for key, val in model_kwargs['y'].items()}

    with torch.no_grad():
        # Using the diffused data from the encoder in the form of noise
        generated_embeddings = generator_sample_fn(
            semantic_generator_model,
            (batch_size, modiffae_latent_dim),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # causes the model to sample xT from a normal distribution
            const_noise=False,
        )

        model_kwargs['y']['semantic_emb'] = generated_embeddings

        modiffae_sample_fn = modiffae_diffusion.ddim_sample_loop

        # Using the diffused data from the encoder in the form of noise
        samples = modiffae_sample_fn(
            modiffae_model,
            (batch_size, data.num_joints, data.num_feats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # causes the model to sample xT from a normal distribution
            const_noise=False,
        )
    return samples, model_kwargs


def generate_samples_for_data_loader(models, n_frames, batch_size, one_hot_labels, modiffae_latent_dim, data):
    generated_samples, model_kwargs = generate_samples(models, n_frames, batch_size,
                                                       one_hot_labels, modiffae_latent_dim, data)
    modiffae_model = models[0][0]
    rot2xyz_pose_rep = data.pose_rep
    rot2xyz_mask = model_kwargs['y']['mask'].reshape(batch_size, n_frames).bool()

    target_label = torch.as_tensor(one_hot_labels[0], dtype=torch.float32).to(dist_util.dev())

    joint_distances = [torch.Tensor(x) for x in data.get_joint_distances()]
    distances = random.choices(joint_distances, k=batch_size)
    distances = collate_tensors(distances)
    distances = distances.to(dist_util.dev())

    generated_samples_xyz = modiffae_model.rot2xyz(x=generated_samples, mask=rot2xyz_mask,
                                                   pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                                   jointstype='karate', vertstrans=True, betas=None, beta=0,
                                                   glob_rot=None, get_rotations_back=False, distance=distances)

    complete_samples = []

    for i in range(generated_samples_xyz.shape[0]):
        sample_xyz = generated_samples_xyz[i]

        sample_xyz = torch.transpose(sample_xyz, 0, 1)
        sample_xyz = torch.transpose(sample_xyz, 0, 2)

        sample_joint_axis_angles, sample_distances = geometry.calc_axis_angles_and_distances(
            points=sample_xyz
        )

        sample_xyz = sample_xyz.cpu().detach().numpy()
        sample_joint_axis_angles = sample_joint_axis_angles.cpu().detach().numpy()
        sample_distances = sample_distances.cpu().detach().numpy()

        target_technique_cls = torch.argmax(target_label[:5]).item()
        target_grade = round(target_label[5].item() * 12)
        target_grade = grade_number_to_name[target_grade]

        complete_sample = (
            sample_xyz,
            sample_joint_axis_angles,
            sample_distances,
            target_technique_cls,
            target_grade
        )
        complete_samples.append(complete_sample)

    j_dist_shape = (len(data_info.reconstruction_skeleton),)
    complete_samples = np.array(complete_samples, dtype=[
            ('joint_positions', 'O'),
            ('joint_axis_angles', 'O'),
            ('joint_distances', 'f4', j_dist_shape),
            ('technique_cls', 'i4'),
            ('grade', 'U10')
        ]
    )
    return complete_samples


def predict_attributes(semantic_regressor_model, generated_samples):
    with torch.no_grad():
        attribute_predictions = semantic_regressor_model(generated_samples)
        action_predictions = attribute_predictions[:, :5]
        action_predictions = F.softmax(action_predictions, dim=-1)
        grade_predictions = attribute_predictions[:, 5]
        grade_predictions = torch.sigmoid(grade_predictions)
        predicted_attributes = torch.cat((action_predictions, grade_predictions.unsqueeze(1)), 1)
    return predicted_attributes


def check_if_prediction_is_correct(prediction, target):
    technique_prediction = torch.argmax(prediction[:5]).item()
    technique_target = torch.argmax(target[:5]).item()

    grade_prediction = prediction[5].item()
    grade_target = target[5].item()

    num_of_grades = 13
    half_distance_between_two_grades = 1 / ((num_of_grades - 1) * 2)

    is_correct = (technique_prediction == technique_target and
                  np.abs(grade_prediction - grade_target) < half_distance_between_two_grades)
    return is_correct


def accepted(prediction, target):
    technique_prediction = torch.argmax(prediction[:5]).item()
    technique_target = torch.argmax(target[:5]).item()
    technique_probability = 1 - (np.abs(technique_prediction - technique_target))

    grade_prediction = prediction[5].item()
    grade_target = target[5].item()
    grade_probability = 1 - (np.abs(grade_prediction - grade_target))

    overall_probability = technique_probability * grade_probability
    random_nr = np.random.rand()
    return overall_probability >= random_nr


def filter_accepted_idxs(predicted_attributes, target_label):
    accepted_idxs = []
    for i in range(predicted_attributes.shape[0]):
        if accepted(predicted_attributes[i], target_label):
            accepted_idxs.append(i)
    return accepted_idxs


def rejection_sampling(models, n_frames, batch_size, one_hot_labels, modiffae_latent_dim, data, joint_distances):
    generated_samples, model_kwargs = generate_samples(models, n_frames, batch_size,
                                                       one_hot_labels, modiffae_latent_dim, data)

    semantic_regressor_model = models[2]
    predicted_attributes = predict_attributes(semantic_regressor_model, generated_samples)

    target_label = torch.as_tensor(one_hot_labels[0], dtype=torch.float32).to(dist_util.dev())

    accepted_sample_idxs = filter_accepted_idxs(predicted_attributes, target_label)

    accepted_samples = generated_samples[accepted_sample_idxs]

    accepted_samples_list = []
    for i in range(accepted_samples.shape[0]):
        modiffae_model = models[0][0]
        rot2xyz_pose_rep = data.pose_rep
        rot2xyz_mask = model_kwargs['y']['mask'].reshape(batch_size, n_frames).bool()

        distances = random.choices(joint_distances, k=batch_size)
        distances = collate_tensors(distances)
        distances = distances.to(dist_util.dev())

        generated_samples_xyz = modiffae_model.rot2xyz(x=generated_samples, mask=rot2xyz_mask,
                                                       pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                                       jointstype='karate', vertstrans=True,  betas=None, beta=0,
                                                       glob_rot=None, get_rotations_back=False, distance=distances)

        accepted_samples_xyz = generated_samples_xyz[accepted_sample_idxs]

        accepted_sample_xyz = accepted_samples_xyz[i]
        accepted_sample_xyz = torch.transpose(accepted_sample_xyz, 0, 1)
        accepted_sample_xyz = torch.transpose(accepted_sample_xyz, 0, 2)

        accepted_sample_joint_axis_angles, accepted_sample_distances = geometry.calc_axis_angles_and_distances(
            points=accepted_sample_xyz
        )

        accepted_sample_xyz = accepted_sample_xyz.cpu().detach().numpy()
        accepted_sample_joint_axis_angles = accepted_sample_joint_axis_angles.cpu().detach().numpy()
        accepted_sample_distances = accepted_sample_distances.cpu().detach().numpy()

        target_technique_cls = torch.argmax(target_label[:5]).item()
        target_grade = round(target_label[5].item() * 12)
        target_grade = grade_number_to_name[target_grade]

        complete_sample = (
            accepted_sample_xyz,
            accepted_sample_joint_axis_angles,
            accepted_sample_distances,
            target_technique_cls,
            target_grade
        )

        accepted_samples_list.append(complete_sample)

    return accepted_samples_list


def main():
    args = generation_args()
    fixseed(args.seed)

    modiffae_args = model_parser(model_type="modiffae", model_path=args.modiffae_model_path)
    test_participant = modiffae_args.test_participant
    data = KaratePoses(test_participant=test_participant, split='train', pose_rep=modiffae_args.pose_rep)
    joint_distances = [torch.Tensor(x) for x in data.get_joint_distances()]

    data_path = os.path.join(data.data_path, f'leave_{modiffae_args.test_participant}_out',
                             f'generated_data_{int(args.ratio * 100)}_percent.npy')
    if os.path.isfile(data_path):
        if not args.overwrite:
            message = 'The target file already exists. If the data should be '
            message += 'replaced by new data, run this script with the --overwrite argument. Exiting...'
            raise Exception(message)

    models = load_models(
        modiffae_model_path=args.modiffae_model_path,
        semantic_generator_model_path=args.semantic_generator_model_path,
        semantic_regressor_model_path=args.semantic_regressor_model_path
    )

    accepted_samples = []

    print(f"Ratio: {args.ratio}")

    number_of_samples_to_generate_per_grade = \
        calc_number_of_samples_to_generate_per_grade(args.ratio, data)

    total_nr_of_samples_to_generate = sum([nr for g, nr in number_of_samples_to_generate_per_grade.items()])
    print(f'Total number of samples to generate: {total_nr_of_samples_to_generate}')

    print(number_of_samples_to_generate_per_grade)

    try:
        for grade, number_of_samples in number_of_samples_to_generate_per_grade.items():
            number_of_samples_per_technique = round(number_of_samples / 5)

            for i in range(5):
                one_hot_labels = create_attribute_labels(grade, i, args.batch_size)

                samples_per_combination = []

                while len(samples_per_combination) < number_of_samples_per_technique:
                    samples = None
                    while samples is None:
                        samples = rejection_sampling(models, args.num_frames, args.batch_size, one_hot_labels,
                                                     modiffae_args.modiffae_latent_dim, data, joint_distances)

                    samples_per_combination.extend(samples)
                    print(f'Accepted sample(s) for grade {grade} and technique {i}')
                    print(f'Progress: {len(accepted_samples) + len(samples_per_combination)}/'
                          f'{total_nr_of_samples_to_generate}')

                accepted_samples.extend(samples_per_combination[:number_of_samples_per_technique])
    except KeyboardInterrupt:
        data_path = os.path.join(data.data_path, f'leave_{modiffae_args.test_participant}_out',
                                 f'generated_data_{int(args.ratio * 100)}_percent_interrupted.npy')

        accepted_samples.extend(samples_per_combination)
        print(len(accepted_samples))
        print('hmm')
    finally:
        nr_generated_samples = len(accepted_samples)
        j_dist_shape = (len(data_info.reconstruction_skeleton),)
        accepted_samples = np.array(accepted_samples, dtype=[
                ('joint_positions', 'O'),
                ('joint_axis_angles', 'O'),
                ('joint_distances', 'f4', j_dist_shape),
                ('technique_cls', 'i4'),
                ('grade', 'U10')
            ]
        )

        np.save(data_path, accepted_samples)
        print(f'Number of generated samples: {nr_generated_samples}')
        print(f'Saved generated data at {data_path}')


if __name__ == "__main__":
    main()
