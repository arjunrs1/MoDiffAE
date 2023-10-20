from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generation_args
from utils.model_util import create_modiffae_and_diffusion, load_model, calculate_embeddings, calculate_z_parameters
from utils import dist_util
from load.get_data import get_dataset_loader
import umap
import matplotlib.pyplot as plt
from model.semantic_regressor import SemanticRegressor
import torch.nn.functional as F
import math
import seaborn as sns
import matplotlib.cm as cm
from itertools import cycle

def normalize(cond):
    std, mean = torch.std_mean(cond)

    cond = (cond - mean.to(dist_util.dev())) / std.to(
        dist_util.dev())
    return cond


def main():
    args = generation_args()

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
    model, diffusion = create_modiffae_and_diffusion(args, train_data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    # Disable random masking
    model.eval()

    cond_mean, cond_std = calculate_z_parameters(train_data, model.semantic_encoder)
    semantic_regressor = SemanticRegressor(
        modiffae_latent_dim=512,
        attribute_dim=6,  # 18,
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

    skill_level_labels = labels[:, 5].tolist()


    #print(technique_labels_arr == 3)
    all_class_idxs = np.arange(0, len(technique_labels))[np.logical_and(technique_labels_arr == 0, np.array(skill_level_labels) < 0.3)]
    #all_class_idxs = np.arange(0, len(technique_labels))[technique_labels_arr == 4]
    #sample_idx = all_class_idxs[32]
    sample_idx = all_class_idxs[0]
    #technique_labels[sample_idx] = 6
    #print(sample_idx)
    modification_sample = semantic_embeddings[sample_idx]

    modifications = []

    """
    for n in np.arange(0.0, 0.5, 0.01):
        #mod = modification_sample - n * math.sqrt(512) * F.normalize(  # 0.9
        #    semantic_regressor.regressor.weight[0][None, :], dim=1)
        mod = modification_sample + n * math.sqrt(512) * F.normalize(  # 0.9
            semantic_regressor.regressor.weight[5][None, :], dim=1)
        modifications.append(mod)
    """

    ### interpolation

    start_emb = semantic_embeddings[5]
    end_emb = semantic_embeddings[50]

    print(start_emb.shape)
    print(end_emb.shape)

    for n in np.arange(0.0, 1.0, 0.05):
        mod = start_emb + (end_emb - start_emb) * n

        print(mod.shape)
        #exit()

        #mod = modification_sample - n * math.sqrt(512) * F.normalize(  # 0.9
        #    semantic_regressor.regressor.weight[0][None, :], dim=1)
        #mod = modification_sample + n * math.sqrt(512) * F.normalize(  # 0.9
        #    semantic_regressor.regressor.weight[5][None, :], dim=1)
        mod = mod.unsqueeze(dim=0)
        modifications.append(mod)

    ####

    modifications = torch.cat(modifications, dim=0)
    # TODO: analyse if this normalization is good
    #modifications = normalize(modifications)

    #modifications = np.array(modifications)

    #mod_class = torch.tensor(5)
    #mod_class = mod_class.repeat(modifications.shape[0])
    mod_class = np.repeat(5, modifications.shape[0])
    #print(modifications.shape)
    #exit()

    print('-----')
    print(semantic_embeddings.shape)
    print(modifications.shape)

    semantic_embeddings = torch.cat((semantic_embeddings, modifications), dim=0)

    semantic_embeddings = semantic_regressor.denormalize(semantic_embeddings)

    semantic_embeddings = semantic_embeddings.cpu().detach().numpy()



    technique_labels.extend(mod_class)


    skill_placeholder = np.repeat(1, modifications.shape[0])
    skill_level_labels.extend(skill_placeholder)

    #print(semantic_embeddings.shape)
    #print(len(technique_labels))
    #exit()



    #skill_level_labels *= 12
    #skill_level_labels = [int(x) for x in skill_level_labels]

    #reducer = umap.UMAP(n_neighbors=100, min_dist=0.5, n_components=2)
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.3, n_components=2)

    embedding = reducer.fit_transform(semantic_embeddings)
    print(embedding.shape)

    #technique_label_names = {
    #    0: 'Gyaku-Zuki',   # reverse punch
    #    1: 'Mae-Geri',   # front kick
    #    2: 'Mawashi-Geri gedan',   # roundhouse kick at knee to hip height
    #    3: 'Mawashi-Geri jodan',   # roundhouse kick at shoulder to (top) head height
    #    4: 'Ushiro-Mawashi-Geri'   # spinning back kick
    #}

    technique_label_names = {
        0: 'Reverse punch',  # reverse punch
        1: 'Front kick',  # front kick
        2: 'Low roundhouse kick',  # roundhouse kick at knee to hip height
        3: 'High roundhouse kick',  # roundhouse kick at shoulder to (top) head height
        4: 'Spinning back kick',  # spinning back kick
        #5: 'Modification'
        5: 'Interpolation'
    }

    colors = cycle(cm.tab10.colors)
    #next(colors)

    #print(np.array(technique_labels) == 2)
    #exit()

    for i in range(6):
        #for _ in range(4):
        color = next(colors)

        idx = np.array(technique_labels) == i
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            #c=[sns.color_palette()[x] for x in np.argmax(technique_labels, axis=1)],
            color=color,
            #s=20,
            label=technique_label_names[i]#,
            #alpha=0.5
        )
    plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(8) - 0.5).set_ticks(np.arange(7))
    #, labels = ["test"]
    #cbar.ax.set_yticklabels(["test"])
    #plt.title('UMAP projection of karate dataset', fontsize=24)
    plt.legend()
    plt.show()

    plt.clf()

    skill_labels = {
        0: "Cluster 1",
        1: "Cluster 2",
        2: "Cluster 3",
        3: "Cluster 4",
        4: "Cluster 1",
        5: "Cluster 2",
        6: "Cluster 3",
        7: "Cluster 4",
        8: "Cluster 1",
        9: "Cluster 2",
        10: "Cluster 3",
        11: "Cluster 4",
        12: "Cluster 1"
    }

    #skill_label_names = [skill_labels[s] for s in skill_level_labels]

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        # c=[sns.color_palette()[x] for x in np.argmax(technique_labels, axis=1)],
        c=skill_level_labels,
        label="test",
        cmap='Blues'
    )
    plt.gca().set_aspect('equal', 'datalim')
    #plt.colorbar().set_ticks(np.arange(13))
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['low', 'high'])

    #plt.title('UMAP projection of karate dataset', fontsize=24)
    plt.show()




if __name__ == "__main__":
    main()
