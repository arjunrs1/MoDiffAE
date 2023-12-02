import math

from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generation_args
from utils.model_util import create_modiffae_and_diffusion, load_model, calculate_z_parameters
from utils import dist_util
from load.get_data import get_dataset_loader
import shutil
from visualize.vicon_visualization import from_array
from model.semantic_regressor import SemanticRegressor
import torch.nn.functional as F

from utils.parser_util import model_parser
from utils.model_util import create_semantic_regressor
from load.tensors import collate, collate_tensors

def load_models(modiffae_model_path, semantic_regressor_model_path):
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

    return (modiffae_model, modiffae_diffusion), semantic_regressor_model


def generate_samples(models, n_frames, batch_size, one_hot_labels, modiffae_latent_dim, data):
    modiffae_model, modiffae_diffusion = models[0]
    #semantic_generator_model, semantic_generator_diffusion = models[1]

    #generator_sample_fn = semantic_generator_diffusion.ddim_sample_loop

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


def normalize(cond):
    std, mean = torch.std_mean(cond)

    cond = (cond - mean.to(dist_util.dev())) / std.to(
        dist_util.dev())
    return cond

def main():
    args = generation_args()

    fixseed(args.seed)
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
                                'reconstruction_samples_{}_{}_seed{}'.format(name, niter, args.seed))

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # We need this check in order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger than default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want more samples, run this script with different seeds
    # (specify through the --seed flag)
    args.num_samples = 10 #10
    args.batch_size = args.num_samples
    args.num_repetitions = 1

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    # Subdirectory for each sample
    for i in range(args.num_samples):
        sample_path = os.path.join(out_path, str(i))
        os.mkdir(sample_path)

    # TODO: make sure that the test set is used

    train_data = load_dataset(args, max_frames, n_frames, split='train')

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames, split='test')
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_modiffae_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    # Disable random masking
    model.eval()

    iterator = iter(data)
    # model_kwargs contains the condition as well as distances
    # for visualization of karate motion.
    data_batch, model_kwargs = next(iterator)
    data_batch = data_batch.to(dist_util.dev())
    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
                         for key, val in model_kwargs['y'].items()}

    # rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
    # rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' \
    #    else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()

    #rot2xyz_pose_rep = model.data_rep
    rot2xyz_pose_rep = model.pose_rep
    rot2xyz_mask = model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()

    distance = model_kwargs['y']['distance']
    og_xyz = model.rot2xyz(x=data_batch, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                           jointstype='karate', vertstrans=True, betas=None, beta=0, glob_rot=None,
                           get_rotations_back=False, distance=distance)
    og_xyz = og_xyz.cpu().numpy()
    lengths = model_kwargs['y']['lengths'].cpu().numpy()

    sample_file_template = 'og_sample{:02d}.ogv'
    for sample_i in range(args.num_samples):
        length = lengths[sample_i]
        m = og_xyz[sample_i].transpose(2, 0, 1)[:length]

        t, j, ax = m.shape

        # It is important that the ordering is correct here.
        # Numpy reshape uses C like indexing by default.
        m = np.reshape(m, (t, j * ax))

        save_file = sample_file_template.format(sample_i)
        animation_save_path = os.path.join(out_path, str(sample_i), save_file)
        from_array(arr=m, sampling_frequency=fps, file_name=animation_save_path)

    all_motions = []
    all_lengths = []
    all_text = []

    # data_batch = data_batch.to(dist_util.dev())
    # cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

    # distance = distance.to(dist_util.dev())

    # for rep_i in range(args.num_repetitions):
    #    print(f'### Sampling [repetitions #{rep_i}]')

    # add CFG scale to batch
    # if args.guidance_param != 1:
    #     model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    diffuse_function = diffusion.ddim_reverse_sample_loop

    print('start reverse ddim')

    # print(args.device)
    # print(data_batch.device)
    # exit()

    # foward pass from encoder
    noise = diffuse_function(
        model,
        data_batch,
        clip_denoised=False,
        model_kwargs=model_kwargs
    )

    # print(noise.keys())
    # print(noise['sample'].shape)

    # exit()

    ##########

    #print(model_kwargs.keys())

    # TODO
    #   - encode semantic
    #   - modify it
    #   - set semantic_emb in model_kwargs
    original_motions = model_kwargs['y']['original_motion']
    with torch.no_grad():
        semantic_embeddings = model.semantic_encoder(original_motions)

    cond_mean, cond_std = calculate_z_parameters(train_data, model.semantic_encoder)
    semantic_regressor = SemanticRegressor(
            modiffae_latent_dim=512,
            attribute_dim=6, #18,
            semantic_encoder=model.semantic_encoder,
            cond_mean=cond_mean,
            cond_std=cond_std
        )

    regressor_path = './save/karateWithValidation/semantic_regressor/model000080000.pt'
    state_dict_reg = torch.load(regressor_path, map_location='cpu')
    load_model(semantic_regressor, state_dict_reg)
    #semantic_regressor.load_state_dict(state_dict_reg, strict=False)
    semantic_regressor.to(dist_util.dev())

    new_semantic_embeddings = semantic_regressor.normalize(semantic_embeddings)

    class_prev = semantic_regressor.regressor(new_semantic_embeddings)
    class_prev = F.sigmoid(class_prev)
    class_prev_technique = F.softmax(class_prev[:, :5])
    print(class_prev_technique)
    #class_prev_skill_level = F.softmax(class_prev[:, 5:])
    class_prev_skill_level = class_prev[:, 5]
    print(class_prev_skill_level)

    print(model_kwargs['y']['labels'])
    #sigmoid_fn = torch.nn.Sigmoid()
    #print(sigmoid_fn(class_prev))
    #print(torch.sum(class_prev, dim=1))

    #print(torch.sigmoid(class_prev))

    print('#####')

    #exit()

    #print(F.normalize(semantic_regressor.regressor.weight[3][None, :], dim=1))

    #exit()
    #new_semantic_embeddings = new_semantic_embeddings + 0.01 * math.sqrt(512) * F.normalize(semantic_regressor.regressor.weight[3][None, :], dim=1) #
    new_semantic_embeddings = new_semantic_embeddings + 0.5 * math.sqrt(512) * F.normalize(  # 0.9
        semantic_regressor.regressor.weight[4][None, :], dim=1)

    # todo: normalize new_semantic_emb again..i think it might be to far off from the distribution
    new_semantic_embeddings = normalize(new_semantic_embeddings)


    class_after = semantic_regressor.regressor(new_semantic_embeddings)
    class_after = F.sigmoid(class_after)
    class_after_technique = F.softmax(class_after[:, :5])
    print(class_after_technique)
    #class_after_skill_level = F.softmax(class_after[:, 5:])
    class_after_skill_level = class_after[:, 5]
    print(class_after_skill_level)

    #print(class_after)
    #print(sigmoid_fn(class_after))
    #print(torch.sum(sigmoid_fn(class_after), dim=1))

    #exit()
    new_semantic_embeddings = semantic_regressor.denormalize(new_semantic_embeddings)



    model_kwargs['y']['semantic_emb'] = new_semantic_embeddings

    print(semantic_embeddings.shape)

    #exit()

    ##########

    print('reverse ddim ran through')
    # This succeeds

    # We do two reconstructions for the same sample.
    # If DDIM is used correctly with eta=0 all decoded
    # motions should be exactly the same.
    for rep_i in range(1):
        print(f'### Sampling [repetitions #{rep_i}]')

        # sample_fn = diffusion.p_sample_loop
        # Anthony: changed to use ddim sampling
        sample_fn = diffusion.ddim_sample_loop

        # Using the diffused data from the encoder in the form of noise
        sample = sample_fn(
            model,
            #(args.batch_size, model.num_joints, model.num_feats, n_frames),
            (args.batch_size, data.dataset.num_joints, data.dataset.num_feats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            # 'sample' should be xT.
            noise=noise['sample'],
            # noise=None,
            const_noise=False,
        )

        print('ddim ran through')

        # Recover XYZ *positions* from HumanML3D vector representation
        # if model.data_rep == 'hml_vec':
        #    n_joints = 22 if sample.shape[1] == 263 else 21
        #    sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        #    sample = recover_from_ric(sample, n_joints)
        #    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        # Modified for karate
        # if args.dataset == 'karate':
        #    j_type = 'karate'
        #    datapath="dataset/KaratePoses"
        #    npydatafilepath = os.path.join(datapath, "karate_motion_25_fps.npy")
        #    all_data = np.load(npydatafilepath, allow_pickle=True)
        #    joint_distances = [x for x in all_data["joint_distances"]]
        #    distance = random.choices(joint_distances, k=args.batch_size)
        # else:
        #    j_type = 'smpl'
        #    distance = None

        # distance = model_kwargs['y']['distance']
        # print(sample.shape)
        # print(len(distance))
        # print(len(distance[0]))

        # print(sample.device)
        # print(distance.device)
        # exit()

        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='karate', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False, distance=distance)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            #text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            #all_text += model_kwargs['y'][text_key]
            pass

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

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
    # if args.dataset == 'karate':
    sample_file_template = 'rec_sample{:02d}_rep{:02d}.ogv'
    for sample_i in range(args.num_samples):
        # rep_files = []
        for rep_i in range(args.num_repetitions):
            # caption = all_text[rep_i*args.batch_size + sample_i]

            # Anthony: I think it might be smart to remove this to allow length change.
            # length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)  # [:length]

            # motion = all_motions[rep_i*args.batch_size + sample_i]
            t, j, ax = motion.shape

            # It is important that the ordering is correct here.
            # Numpy reshape uses C like indexing by default.
            motion = np.reshape(motion, (t, j * ax))

            #print(motion)

            #print(motion.shape)

            save_file = sample_file_template.format(sample_i, rep_i)
            # print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, str(sample_i), save_file)
            from_array(arr=motion, sampling_frequency=fps, file_name=animation_save_path)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def load_dataset(args, max_frames, n_frames, split='test'):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              test_participant='b0372',
                              #split='test')
                              split=split)
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
