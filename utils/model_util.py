from model.modiffae import MoDiffAE
from model.semantic_generator import SemanticGenerator
from model.semantic_regressor import SemanticRegressor
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
import torch
from utils import dist_util


def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0


def create_modiffae_and_diffusion(args, data):
    #model = MoDiffAE(**get_model_args(args, data))
    model = MoDiffAE(
        num_joints=data.dataset.num_joints,
        num_feats=data.dataset.num_feats,
        num_frames=args.num_frames,
        pose_rep=args.pose_rep,
        translation=(not args.no_translation),
        modiffae_latent_dim=args.modiffae_latent_dim,
        transformer_feedforward_dim=args.transformer_feedforward_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        semantic_pool_type=args.semantic_pool_type,
        dataset=data.dataset.data_name
    )
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def create_semantic_generator_and_diffusion(args):
    diffusion = create_gaussian_diffusion(args)
    model = SemanticGenerator(
        attribute_dim=args.attribute_dim,
        modiffae_latent_dim=args.modiffae_latent_dim,
        latent_dim=args.semantic_generator_latent_dim,
        num_layers=args.layers,
        dropout=args.dropout
    )
    return model, diffusion


def create_semantic_regressor(args, train_data, semantic_encoder):
    cond_mean, cond_std = calculate_z_parameters(train_data, semantic_encoder)
    model = SemanticRegressor(
        modiffae_latent_dim=args.modiffae_latent_dim,
        attribute_dim=args.attribute_dim,
        semantic_encoder=semantic_encoder,
        cond_mean=cond_mean,
        cond_std=cond_std
    )
    return model


'''def get_model_args(args, data):
    return {'num_joints': data.dataset.num_joints, 'num_feats': data.dataset.num_feats,
            'num_frames': args.num_frames, 'pose_rep': data.dataset.pose_rep, 'translation': True,
            'modiffae_latent_dim': args.modiffae_latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data.dataset.pose_rep, #'cond_mode': cond_mode,
            #'cond_mask_prob': args.cond_mask_prob,
            'dataset': args.dataset}'''


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    #steps = 1000  # args.diffusion_steps
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    #timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    # Anthony: Skipping no steps
    #timestep_respacing = 'ddim1000'
    timestep_respacing = f'ddim{steps}'
    learn_sigma = False
    rescale_timesteps = False

    print(f'Number of diffusion steps: {steps}')

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    try:
        lambda_vel = args.lambda_vel
        lambda_rcxyz = args.lambda_rcxyz
        lambda_fc = args.lambda_fc
    except AttributeError:
        lambda_vel = 0
        lambda_rcxyz = 0
        lambda_fc = 0

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=lambda_vel,
        lambda_rcxyz=lambda_rcxyz,
        lambda_fc=lambda_fc,
        #lambda_vel=args.lambda_vel,
        #lambda_rcxyz=args.lambda_rcxyz,
        #lambda_fc=args.lambda_fc,
    )


def calculate_embeddings(data, semantic_encoder, return_labels=False):
    embeddings = []
    labels = []
    for motion, cond in data:
        cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                     cond['y'].items()}
        og_motion = cond['y']['original_motion']
        batch_labels = cond['y']['labels']
        batch_labels = batch_labels.squeeze()

        with torch.no_grad():
            emb = semantic_encoder(og_motion)
        embeddings.append(emb)
        labels.append(batch_labels)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    if return_labels:
        return embeddings, labels
    else:
        return embeddings


#def calculate_embeddings_from_data

def calculate_z_parameters(data, semantic_encoder, embeddings=None):
    if embeddings is None:
        embeddings = calculate_embeddings(data, semantic_encoder)
    #embeddings = torch.cat(embeddings)
    std, mean = torch.std_mean(embeddings)
    return mean, std
