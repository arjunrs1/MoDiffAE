from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model, calculate_embeddings, calculate_z_parameters
from utils import dist_util
from load.get_data import get_dataset_loader
import umap
import matplotlib.pyplot as plt
from model.semantic_regressor import SemanticRegressor
import torch.nn.functional as F
import math
import seaborn as sns


def normalize(cond):
    std, mean = torch.std_mean(cond)

    cond = (cond - mean.to(dist_util.dev())) / std.to(
        dist_util.dev())
    return cond


def main():
    args = generate_args()

    fixseed(args.seed)
    #out_path = args.output_dir

    dist_util.setup_dist(args.device)

    print('Loading dataset...')
    train_data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        test_participant='b0372',
        split='train'
    )

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, train_data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    # Disable random masking
    model.eval()

    cond_mean, cond_std = calculate_z_parameters(train_data, model.semantic_encoder)
    semantic_regressor = SemanticRegressor(
        input_dim=512,
        output_dim=6,  # 18,
        semantic_encoder=model.semantic_encoder,
        cond_mean=cond_mean,
        cond_std=cond_std
    )

    regressor_path = './save/karateOnlyAir/semantic_regressor/semantic_regressor/model000050000.pt'
    state_dict_reg = torch.load(regressor_path, map_location='cpu')
    load_model(semantic_regressor, state_dict_reg)
    # semantic_regressor.load_state_dict(state_dict_reg, strict=False)
    semantic_regressor.to(dist_util.dev())

    semantic_embeddings, labels = calculate_embeddings(train_data, model.semantic_encoder, return_labels=True)
    semantic_embeddings = semantic_regressor.normalize(semantic_embeddings)

    labels = labels.cpu().detach().numpy()

    technique_labels = labels[:, :5]
    technique_labels = [x for x in np.argmax(technique_labels, axis=1)]

    print(semantic_embeddings.shape)

    print(technique_labels)

    technique_labels_arr = np.array(technique_labels)
    #print(technique_labels_arr == 3)
    all_class_idxs = np.arange(0, len(technique_labels))[technique_labels_arr == 4]
    sample_idx = all_class_idxs[32]
    technique_labels[sample_idx] = 6
    #print(sample_idx)
    modification_sample = semantic_embeddings[sample_idx]

    modifications = []

    for n in np.arange(0.0, 0.5, 0.01):
        mod = modification_sample - n * math.sqrt(512) * F.normalize(  # 0.9
            semantic_regressor.regressor.weight[0][None, :], dim=1)
        modifications.append(mod)

    modifications = torch.cat(modifications)
    # TODO: analyse if this normalization is good
    #modifications = normalize(modifications)

    #mod_class = torch.tensor(5)
    #mod_class = mod_class.repeat(modifications.shape[0])
    mod_class = np.repeat(5, modifications.shape[0])
    #print(modifications.shape)
    #exit()

    semantic_embeddings = torch.cat((semantic_embeddings, modifications), dim=0)

    semantic_embeddings = semantic_regressor.denormalize(semantic_embeddings)

    semantic_embeddings = semantic_embeddings.cpu().detach().numpy()



    technique_labels.extend(mod_class)

    #print(semantic_embeddings.shape)
    #print(len(technique_labels))
    #exit()

    skill_level_labels = labels[:, 5]

    skill_level_labels *= 12
    skill_level_labels = [int(x) for x in skill_level_labels]

    #reducer = umap.UMAP(n_neighbors=100, min_dist=0.5, n_components=2)
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.3, n_components=2)

    embedding = reducer.fit_transform(semantic_embeddings)
    print(embedding.shape)

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        #c=[sns.color_palette()[x] for x in np.argmax(technique_labels, axis=1)],
        c=technique_labels,
        cmap='Spectral'
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(8) - 0.5).set_ticks(np.arange(7))
    plt.title('UMAP projection of karate dataset', fontsize=24)
    plt.show()

    '''plt.clf()

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        # c=[sns.color_palette()[x] for x in np.argmax(technique_labels, axis=1)],
        c=skill_level_labels,
        cmap='Spectral'
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(614) - 0.5).set_ticks(np.arange(13))
    plt.title('UMAP projection of karate dataset', fontsize=24)
    plt.show()'''


if __name__ == "__main__":
    main()
