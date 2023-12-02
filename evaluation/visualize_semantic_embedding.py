from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import evaluation_args, model_parser
from utils.model_util import create_modiffae_and_diffusion, load_model, calculate_embeddings
from utils import dist_util
from load.get_data import get_dataset_loader
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle
from utils.karate.data_info import technique_class_to_name


def main():
    args = evaluation_args()

    modiffae_args = model_parser(model_type="modiffae")

    fixseed(args.seed)

    _, model_name = os.path.split(args.modiffae_model_path)
    model_name = model_name.split('.')[0]

    save_dir = os.path.join(os.path.dirname(args.modiffae_model_path), 'evaluation', model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dist_util.setup_dist(args.device)

    print('Loading dataset...')
    train_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='train'
    )

    print("Creating model and diffusion...")
    model, diffusion = create_modiffae_and_diffusion(modiffae_args, train_data)

    print(f"Loading checkpoints from [{args.modiffae_model_path}]...")
    state_dict = torch.load(args.modiffae_model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    model.eval()

    semantic_embeddings, labels = calculate_embeddings(train_data, model.semantic_encoder, return_labels=True)

    labels = labels.cpu().detach().numpy()

    technique_labels = labels[:, :5]
    technique_labels = [x for x in np.argmax(technique_labels, axis=1)]

    skill_level_labels = labels[:, 5].tolist()

    semantic_embeddings = semantic_embeddings.cpu().detach().numpy()

    # reducer = umap.UMAP(n_neighbors=100, min_dist=0.5, n_components=2)
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.3, n_components=2)

    embedding = reducer.fit_transform(semantic_embeddings)

    colors = cycle(cm.tab10.colors)

    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(15)

    for i in technique_class_to_name.keys():
        color = next(colors)

        idx = np.array(technique_labels) == i
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            color=color,
            label=technique_class_to_name[i]
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.legend()
    grade_plot_path = os.path.join(save_dir, 'grade_embedding.png')
    if os.path.isfile(grade_plot_path) and not args.overwrite:
        raise FileExistsError('File [{}] already exists.'.format(grade_plot_path))
    else:
        plt.savefig(grade_plot_path)
    #plt.show()
    plt.clf()

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=skill_level_labels,
        cmap='Blues'
    )
    plt.gca().set_aspect('equal', 'datalim')
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['low grade', 'high grade'])
    technique_plot_path = os.path.join(save_dir, 'technique_embedding.png')
    if os.path.isfile(technique_plot_path) and not args.overwrite:
        raise FileExistsError('File [{}] already exists.'.format(technique_plot_path))
    else:
        plt.savefig(technique_plot_path)
    #plt.show()


if __name__ == "__main__":
    main()
