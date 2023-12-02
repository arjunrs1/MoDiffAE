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
            # (args.batch_size, model.num_joints, model.num_feats, n_frames),
            (batch_size, modiffae_latent_dim),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # causes the model to sample xT from a normal distribution
            # noise=None,
            const_noise=False,
        )

        model_kwargs['y']['semantic_emb'] = generated_embeddings

        # sample_fn = diffusion.p_sample_loop
        # Anthony: changed to use ddim sampling. Maybe use ddpm function instead
        modiffae_sample_fn = modiffae_diffusion.ddim_sample_loop

        # Using the diffused data from the encoder in the form of noise
        samples = modiffae_sample_fn(
            modiffae_model,
            # (args.batch_size, model.num_joints, model.num_feats, n_frames),
            (batch_size, data.num_joints, data.num_feats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # causes the model to sample xT from a normal distribution
            # noise=None,
            const_noise=False,
        )
    return samples, model_kwargs


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


def rejection_sampling(models, n_frames, batch_size, one_hot_labels, modiffae_latent_dim, data, joint_distances):
    generated_samples, model_kwargs = generate_samples(models, n_frames, batch_size,
                                                       one_hot_labels, modiffae_latent_dim, data)

    semantic_regressor_model = models[2]
    predicted_attributes = predict_attributes(semantic_regressor_model, generated_samples)

    target_label = torch.as_tensor(one_hot_labels[0], dtype=torch.float32).to(dist_util.dev())
    prediction_distances = torch.abs(predicted_attributes - target_label)
    # Weighing so that the influence of grade and technique is equal
    prediction_distances[:, 5] *= 5
    weighted_average_distance = torch.mean(prediction_distances, dim=1)
    idx_of_closest = torch.argmin(weighted_average_distance)
    closest_prediction = predicted_attributes[idx_of_closest]

    if check_if_prediction_is_correct(closest_prediction, target_label):
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
        accepted_sample_xyz = generated_samples_xyz[idx_of_closest]
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
        return complete_sample
    else:
        return None


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

    number_of_samples_to_generate_per_grade = \
        calc_number_of_samples_to_generate_per_grade(args.ratio, data)

    total_nr_of_samples_to_generate = sum([nr for g, nr in number_of_samples_to_generate_per_grade.items()])
    print(f'Total number of samples to generate: {total_nr_of_samples_to_generate}')

    print(number_of_samples_to_generate_per_grade)

    for grade, number_of_samples in number_of_samples_to_generate_per_grade.items():
        number_of_samples_per_technique = round(number_of_samples / 5)
        for i in range(5):
            one_hot_labels = create_attribute_labels(grade, i, args.batch_size)

            generation_count = 0
            while generation_count < number_of_samples_per_technique:
                sample = None
                while sample is None:
                    sample = rejection_sampling(models, args.num_frames, args.batch_size, one_hot_labels,
                                                modiffae_args.modiffae_latent_dim, data, joint_distances)
                accepted_samples.append(sample)
                print(f'Accepted a sample for grade {grade} and technique {i}')
                print(f'Progress: {len(accepted_samples)}/{total_nr_of_samples_to_generate}')
                generation_count += 1

                '''break
            break
        break'''

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



    #print(number_of_samples_to_generate_per_grade)

    """load_models(
        modiffae_model_path="./save/rot_6d_karate/modiffae_b0372/model000300000.pt",
        semantic_generator_model_path="./save/rot_6d_karate/semantic_generator_based_on_modiffae_b0372_model000300000/model000000000.pt",
        semantic_regressor_model_path="./save/rot_6d_karate/semantic_regressor_based_on_modiffae_b0372_model000300000/model000000000.pt"
    )"""
    #exit()

    #args = generation_args()

    """fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    if args.dataset == 'karate':
        max_frames = 100
    else:
        raise NotImplementedError("No number of maximum frames specified for this dataset.")

    if args.dataset == 'karate':
        fps = 25
    else:
        raise NotImplementedError("No framerate specified for this dataset.")
    n_frames = min(max_frames, int(args.motion_length * fps))

    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'generated_samples_{}_{}_seed{}'.format(name, niter, args.seed))

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # We need this check in order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger than default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want more samples, run this script with different seeds
    # (specify through the --seed flag)
    args.num_samples = 10  # 10
    args.batch_size = args.num_samples
    #args.num_repetitions = 1

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    # Subdirectory for each sample
    for i in range(args.num_samples):
        sample_path = os.path.join(out_path, str(i))
        os.mkdir(sample_path)

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames, split='test')
    #total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_modiffae_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    # Disable random masking
    model.eval()

    args.latent_model_path = "./save/karateWithValidation/latentNet_ok/model000120000.pt"

    emb_model, emb_diffusion = create_semantic_generator_and_diffusion(args)
    print(f"Loading checkpoints from [{args.latent_model_path}]...")
    latent_state_dict = torch.load(args.latent_model_path, map_location='cpu')
    load_model(emb_model, latent_state_dict)
    emb_model.to(dist_util.dev())
    emb_model.eval()
    # TODO: generate z (shape as the batch size)
    generator_sample_fn = diffusion.ddim_sample_loop

    ####

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
    #is_t2m = any([args.input_text, args.text_prompt])
    #if is_t2m:
        # t2m
    #    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    #else:
        # a2m
        #action = data.dataset.action_name_to_action(action_text)

    #label_tensors =

    skill_labels = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
    # one_hot_skill_labels = np.eye(len(karate_grade_enumerator))[skill_labels]

    labels = np.array([3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
    # TODO: check if this works
    one_hot_labels = np.eye(5)[labels]

    #print(one_hot_labels)
    #print(skill_labels)
    #exit()



    # one_hot_labels = np.append(one_hot_labels, one_hot_skill_labels, axis=1)
    one_hot_labels = np.append(one_hot_labels, skill_labels, axis=1)

    collate_args = [dict(arg, labels=l) for
                    arg, l in zip(collate_args, one_hot_labels)]
    _, model_kwargs = collate(collate_args)

    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
                         for key, val in model_kwargs['y'].items()}

    print(model_kwargs)


    ####

    args.emb_dim = 512

    # Using the diffused data from the encoder in the form of noise
    generator_samples = generator_sample_fn(
        emb_model,
        # (args.batch_size, model.num_joints, model.num_feats, n_frames),
        (args.batch_size, args.emb_dim),  # TODO: check if emb_dim is correct
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,  # causes the model to sample xT from a normal distribution
        # noise=None,
        const_noise=False,
    )

    #exit()

    rot2xyz_pose_rep = model.pose_rep
    rot2xyz_mask = model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()



    model_kwargs['y']['semantic_emb'] = generator_samples

    # sample_fn = diffusion.p_sample_loop
    # Anthony: changed to use ddim sampling. Maybe use ddpm function instead
    sample_fn = diffusion.ddim_sample_loop

    # Using the diffused data from the encoder in the form of noise
    samples = sample_fn(
        model,
        # (args.batch_size, model.num_joints, model.num_feats, n_frames),
        (args.batch_size, data.dataset.num_joints, data.dataset.num_feats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,  # causes the model to sample xT from a normal distribution
        # noise=None,
        const_noise=False,
    )

    print(samples.shape)
    #exit()

    # Modified for karate
    if args.dataset == 'karate':
        j_type = 'karate'
        datapath = "datasets/kyokushin_karate"
        npydatafilepath = os.path.join(datapath, "karate_motion_modified.npy")
        all_data = np.load(npydatafilepath, allow_pickle=True)
        joint_distances = [torch.Tensor(x) for x in all_data["joint_distances"]]
        distance = random.choices(joint_distances, k=args.batch_size)
        distance = collate_tensors(distance)
        distance = distance.to(dist_util.dev())
    else:
        j_type = 'smpl'
        distance = None

    samples = model.rot2xyz(x=samples, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                           jointstype='karate', vertstrans=True, betas=None, beta=0, glob_rot=None,
                           get_rotations_back=False, distance=distance)

    print(samples.shape)

    # TODO: visualize and save

    sample_file_template = 'genertaed_sample{:02d}.ogv'

    for sample_i in range(args.num_samples):
        # rep_files = []
        #for rep_i in range(args.num_repetitions):
            # caption = all_text[rep_i*args.batch_size + sample_i]

            # Anthony: I think it might be smart to remove this to allow length change.
            # length = all_lengths[rep_i*args.batch_size + sample_i]
            #motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)  # [:length]
        s = samples[sample_i, :, :, :].cpu().numpy()
        #print(s.shape)
        #exit()
        motion = s.transpose(2, 0, 1)  # [:length]

        # motion = all_motions[rep_i*args.batch_size + sample_i]
        t, j, ax = motion.shape

        # It is important that the ordering is correct here.
        # Numpy reshape uses C like indexing by default.
        motion = np.reshape(motion, (t, j * ax))

        #print(motion)

        #print(motion.shape)

        #save_file = sample_file_template.format(sample_i, rep_i)
        # print(sample_print_template.format(caption, sample_i, rep_i, save_file))
        #animation_save_path = os.path.join(out_path, str(sample_i), save_file)
        #from_array(arr=motion, sampling_frequency=fps, file_name=animation_save_path)
        from_array(arr=motion, sampling_frequency=fps)  # , file_name=animation_save_path)


    exit()"""

    ################

    """
    args = generate_args()
    #print(args.dataset)

    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    # Added karate
    if args.dataset == 'karate': 
        max_frames = 100   #125 # 250 # (125 at 25 fps means max 5 seconds)
    else:
        max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    # Added karate 
    if args.dataset == 'karate': 
        fps = 25 # 50
    else:
        fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger than default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_modiffae_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    if args.guidance_param != 1:
        print('Using classifier-free guidance')
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        #sample_fn = diffusion.p_sample_loop
        # Anthony: changed to use ddim sampling 

        sample_fn = diffusion.ddim_sample_loop

        samples = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        #if model.data_rep == 'hml_vec':
        #    n_joints = 22 if sample.shape[1] == 263 else 21
        #    sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        #    sample = recover_from_ric(sample, n_joints)
        #    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        
        # Modified for karate
        if args.dataset == 'karate': 
            j_type = 'karate'
            datapath="datasets/karate"
            npydatafilepath = os.path.join(datapath, "karate_motion_modified.npy")
            all_data = np.load(npydatafilepath, allow_pickle=True)
            joint_distances = [torch.Tensor(x) for x in all_data["joint_distances"]]
            distance = random.choices(joint_distances, k=args.batch_size)
            distance = collate_tensors(distance)
            distance = distance.to(dist_util.dev())
        else:
            j_type = 'smpl'
            distance = None

        print(samples.device)
        print(distance.device)
        
        samples = model.rot2xyz(x=samples, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                            jointstype=j_type, vertstrans=True, betas=None, beta=0, glob_rot=None,
                            get_rotations_back=False, distance=distance)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(samples.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
            'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")

    # Added karate 
    if args.dataset == 'karate':
        sample_file_template = 'sample{:02d}_rep{:02d}.ogv'
        for sample_i in range(args.num_samples):
            #rep_files = []
            for rep_i in range(args.num_repetitions):
                #caption = all_text[rep_i*args.batch_size + sample_i]
                length = all_lengths[rep_i*args.batch_size + sample_i]
                motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                
                #motion = all_motions[rep_i*args.batch_size + sample_i]
                t, j, ax = motion.shape
                
                # It is important that the ordering is correct here.
                # Numpy reshape uses C like indexing by default.
                motion = np.reshape(motion, (t, j*ax))

                save_file = sample_file_template.format(sample_i, rep_i)
                #print(sample_print_template.format(caption, sample_i, rep_i, save_file))
                animation_save_path = os.path.join(out_path, save_file)
                from_array(arr=motion, sampling_frequency=fps, file_name=animation_save_path)
    else:
        pass
        '''
        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

        sample_files = []
        num_samples_in_out_file = 7

        sample_print_template, row_print_template, all_print_template, \
        sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

        for sample_i in range(args.num_samples):
            rep_files = []
            for rep_i in range(args.num_repetitions):
                caption = all_text[rep_i*args.batch_size + sample_i]
                length = all_lengths[rep_i*args.batch_size + sample_i]
                motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                save_file = sample_file_template.format(sample_i, rep_i)
                print(sample_print_template.format(caption, sample_i, rep_i, save_file))
                animation_save_path = os.path.join(out_path, save_file)
                plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
                # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
                rep_files.append(animation_save_path)

            sample_files = save_multiple_samples(args, out_path,
                                                row_print_template, all_print_template, row_file_template, all_file_template,
                                                caption, num_samples_in_out_file, rep_files, sample_files, sample_i)
        '''

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template"""

"""def load_dataset(args, max_frames, n_frames, split='test'):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              test_participant='b0372',
                              #split='test')
                              split=split)
    data.fixed_length = n_frames
    return data"""

"""def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              test_participant='b0372',
                              split='test') #,
                              #hml_mode='text_only')
    data.fixed_length = n_frames
    return data"""


if __name__ == "__main__":
    main()
