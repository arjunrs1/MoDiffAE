from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import model_parser, modiffae_validation_args
from utils.model_util import create_modiffae_and_diffusion, load_model, calculate_embeddings
from utils import dist_util
from load.get_data import get_dataset_loader
import matplotlib.pyplot as plt


def calc_distance_score(train_embeddings, train_labels, validation_embedding, validation_label, k):
    distances_to_train_embeddings_list = list(np.linalg.norm(train_embeddings - validation_embedding, axis=1))
    train_labels_list = list(train_labels)

    distances_and_labels = list(zip(distances_to_train_embeddings_list, train_labels_list))
    distances_and_labels.sort(key=lambda x: x[0])

    k_closest = distances_and_labels[:k]
    k_closest_labels = np.array([e[1] for e in k_closest])

    k_closest_technique_labels = k_closest_labels[:, :5]
    k_closest_technique_labels_cls = [x for x in np.argmax(k_closest_technique_labels, axis=1)]

    k_closest_average_label = np.mean(k_closest_labels, axis=0)
    k_closest_average_label = np.expand_dims(k_closest_average_label, axis=1)

    val_label = np.expand_dims(validation_label, axis=1)
    val_label_cls = np.argmax(val_label)

    acc = k_closest_technique_labels_cls.count(val_label_cls) / len(k_closest_technique_labels_cls)

    label_distance = np.linalg.norm(k_closest_average_label - val_label, axis=1)
    skill_mae = label_distance[-1]

    errors = [(val_label_cls, acc), skill_mae]

    return errors


def calc_checkpoint_metrics(model_path):
    modiffae_args = model_parser(model_type="modiffae", model_path=model_path)

    train_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='train'
    )

    validation_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='validation'
    )

    model, diffusion = create_modiffae_and_diffusion(modiffae_args, train_data)

    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    model.eval()

    train_semantic_embeddings, train_labels = calculate_embeddings(train_data, model.semantic_encoder,
                                                                   return_labels=True)
    train_semantic_embeddings = train_semantic_embeddings.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy()

    validation_semantic_embeddings, validation_labels = calculate_embeddings(validation_data, model.semantic_encoder,
                                                                             return_labels=True)
    validation_semantic_embeddings = validation_semantic_embeddings.cpu().detach().numpy()
    validation_labels = validation_labels.cpu().detach().numpy()

    error_scores = []
    for i in range(validation_semantic_embeddings.shape[0]):
        val_embedding = validation_semantic_embeddings[i]
        val_label = validation_labels[i]
        error_score = calc_distance_score(train_semantic_embeddings, train_labels, val_embedding, val_label, k=50)
        error_scores.append(error_score)

    technique_accuracies = []
    for cls in range(5):
        tech_scores = [ac for (c, ac), _ in error_scores if c == cls]
        tech_scores_avg = np.mean(tech_scores)
        technique_accuracies.append(tech_scores_avg)

    skill_mae = np.mean([err for _, err in error_scores])

    metrics = np.append(technique_accuracies, skill_mae)
    return metrics


def main():
    args = modiffae_validation_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    checkpoints = [p for p in sorted(os.listdir(args.save_dir)) if p.startswith('model') and p.endswith('.pt')]

    checkpoint_metrics = []
    for ch in checkpoints:
        model_path = os.path.join(args.save_dir, ch)
        checkpoint_metric = calc_checkpoint_metrics(model_path)
        checkpoint_metrics.append(checkpoint_metric)

    checkpoint_metrics = np.array(checkpoint_metrics)

    checkpoints = [str(int(int(ch.strip("model").strip(".pt")) / 1000)) + "k" for ch in checkpoints]

    idx_to_name = {
        0: "Acc: Reverse punch",
        1: "Acc: Front kick",
        2: "Acc: Low roundhouse kick",
        3: "Acc: High roundhouse kick",
        4: "Acc: Spinning back kick",
        5: "Mae: Grade"
    }

    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(8)

    x = checkpoints
    for idx in range(checkpoint_metrics.shape[1]):
        y = checkpoint_metrics[:, idx]
        plt.plot(x, y, label=f"{idx_to_name[idx]}")

    # TODO: think about metric at which to stop. Some sort of weighted average
    #       maybe 1 - skill_dist averaged with accuracies but both equally weighted

    plt.legend()

    eval_dir = os.path.join(args.save_dir, "evaluation")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    fig_save_path = os.path.join(eval_dir, "knn_metrics")

    plt.savefig(fig_save_path)


if __name__ == "__main__":
    main()
