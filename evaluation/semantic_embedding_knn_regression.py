from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import model_parser, modiffae_validation_args
from utils.model_util import create_modiffae_and_diffusion, load_model, calculate_embeddings
from utils import dist_util
from load.get_data import get_dataset_loader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
import seaborn as sns
import colorcet as cc
import pandas as pd
import json


def calc_distance_score(train_embeddings, train_labels, validation_embedding,
                        validation_label, grade_prio_probabilities, k):
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

    technique_prediction = max(k_closest_technique_labels_cls, key=k_closest_technique_labels_cls.count)

    technique_target = np.argmax(validation_label[:5])

    val_label = np.expand_dims(validation_label, axis=1)

    if technique_prediction == technique_target:
        technique_acc = 1
    else:
        technique_acc = 0

    grade_mae = np.linalg.norm(k_closest_average_label[5] - val_label[5])

    grade_prediction = round(np.squeeze(k_closest_average_label[5]) * 12)
    grade_target_float = validation_label[5]
    grade_target = round(grade_target_float * 12)

    errors = [(technique_target, technique_acc), (grade_target, grade_mae)]

    predictions_and_targets = (
        (technique_prediction, technique_target),
        (grade_prediction, grade_target)
    )

    return errors, predictions_and_targets


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

    grade_train_labels = list(train_labels[:, 5])
    grade_train_labels = [round(lab * 12) for lab in grade_train_labels]
    grade_prio_probabilities = {gr: cnt/len(grade_train_labels) for gr, cnt in Counter(grade_train_labels).items()}

    validation_semantic_embeddings, validation_labels = calculate_embeddings(validation_data, model.semantic_encoder,
                                                                             return_labels=True)
    validation_semantic_embeddings = validation_semantic_embeddings.cpu().detach().numpy()
    validation_labels = validation_labels.cpu().detach().numpy()

    technique_predictions = []
    technique_targets = []
    grade_predictions = []
    grade_targets = []

    error_scores = []
    for i in range(validation_semantic_embeddings.shape[0]):
        val_embedding = validation_semantic_embeddings[i]
        val_label = validation_labels[i]
        error_score, predictions_and_targets = (
            calc_distance_score(train_semantic_embeddings, train_labels,
                                val_embedding, val_label, grade_prio_probabilities, k=15))
        error_scores.append(error_score)

        technique_predictions.append(predictions_and_targets[0][0])
        technique_targets.append(predictions_and_targets[0][1])
        grade_predictions.append(predictions_and_targets[1][0])
        grade_targets.append(predictions_and_targets[1][1])

    predictions_and_targets_combined = (
        (technique_predictions, technique_targets),
        (grade_predictions, grade_targets)
    )

    technique_accuracies = []
    for cls in range(5):
        tech_scores = [ac for (c, ac), _ in error_scores if c == cls]
        tech_scores_avg = np.mean(tech_scores)
        technique_accuracies.append(tech_scores_avg)

    grade_maes = []
    for gr in range(13):
        grade_scores = [mae for _, (g, mae) in error_scores if g == gr]
        grade_scores_avg = np.mean(grade_scores)
        grade_maes.append(grade_scores_avg)

    return technique_accuracies, grade_maes, predictions_and_targets_combined


def main():
    args = modiffae_validation_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    checkpoints = [p for p in sorted(os.listdir(args.save_dir)) if p.startswith('model') and p.endswith('.pt')]

    test_participant = '_b0' + args.save_dir.split('b0')[-1].split('/')[0]

    technique_accuracies_all = []
    grade_maes_all = []
    predictions_and_targets_all = []
    for ch in checkpoints:
        model_path = os.path.join(args.save_dir, ch)
        technique_accuracies, grade_maes, predictions_and_targets_combined = calc_checkpoint_metrics(model_path)
        technique_accuracies_all.append(technique_accuracies)
        grade_maes_all.append(grade_maes)
        predictions_and_targets_all.append(predictions_and_targets_combined)

    technique_accuracies_all = np.array(technique_accuracies_all)
    grade_maes_all = np.array(grade_maes_all)

    checkpoints = [str(int(int(ch.strip("model").strip(".pt")) / 1000)) + "K" for ch in checkpoints]

    technique_idx_to_name = {
        0: "ACC: Reverse punch",
        1: "ACC: Front kick",
        2: "ACC: Low roundhouse kick",
        3: "ACC: High roundhouse kick",
        4: "ACC: Spinning back kick"
    }

    technique_idx_to_name_short = {
        0: "RP",
        1: "FK",
        2: "LRK",
        3: "HRK",
        4: "SBK"
    }

    grade_idx_to_name = {
        0: 'MAE: 9 kyu',
        1: 'MAE: 8 kyu',
        2: 'MAE: 7 kyu',
        3: 'MAE: 6 kyu',
        4: 'MAE: 5 kyu',
        5: 'MAE: 4 kyu',
        6: 'MAE: 3 kyu',
        7: 'MAE: 2 kyu',
        8: 'MAE: 1 kyu',
        9: 'MAE: 1 dan',
        10: 'MAE: 2 dan',
        11: 'MAE: 3 dan',
        12: 'MAE: 4 dan'
    }

    grade_idx_to_name_short = {
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

    f = plt.figure()
    f.set_figwidth(18)
    f.set_figheight(8)

    plt.rc('font', size=20)
    plt.rc('legend', fontsize=16)

    plt.rc('legend', fontsize=14)

    x = checkpoints

    x[-1] = "1M"

    for idx in range(technique_accuracies_all.shape[1]):
        y = technique_accuracies_all[:, idx]
        plt.plot(x, y, label=f"{technique_idx_to_name[idx]}")

    technique_unweighted_average_recalls = []
    for idx in range(technique_accuracies_all.shape[0]):
        technique_unweighted_average_recalls.append(np.mean(technique_accuracies_all[idx, :]))
    best_technique_avg_idx = np.argmax(technique_unweighted_average_recalls)

    plt.plot(x, technique_unweighted_average_recalls, label=f"UAR", color='black')

    plt.vlines(x=[best_technique_avg_idx], ymin=0, ymax=1, colors='black', ls='--', lw=2,
               label='Best UAR')

    plt.legend()
    plt.xlabel('Training steps')

    plt.xticks([x if i % 2 != 1 else '' for i, x in enumerate(x)])

    eval_dir = os.path.join(args.save_dir, "evaluation")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    fig_save_path = os.path.join(eval_dir, f"knn_technique_uar_{test_participant}")
    plt.savefig(fig_save_path)

    plt.clf()

    plt.rc('legend', fontsize=12)

    with sns.color_palette(cc.glasbey, n_colors=14):
        for idx in reversed(list(range(grade_maes_all.shape[1]))):
            y = grade_maes_all[:, idx]
            plt.plot(x, y, label=f"{grade_idx_to_name[idx]}")

    grade_averages = []
    for idx in range(grade_maes_all.shape[0]):
        grade_averages.append(np.mean(grade_maes_all[idx, :]))
    best_grade_avg_idx = np.argmin(grade_averages)

    plt.plot(x, grade_averages, label=f"UMAE", color='black')

    plt.vlines(x=[best_grade_avg_idx], ymin=0, ymax=1, colors='black', ls='--', lw=2,
               label='Best UMAE')

    plt.legend()

    plt.xticks([x if i % 2 != 1 else '' for i, x in enumerate(x)])
    plt.xlabel('Training steps')

    fig_save_path = os.path.join(eval_dir, f"knn_grade_umae_{test_participant}")
    plt.savefig(fig_save_path)

    plt.clf()

    f = plt.figure()
    f.set_figwidth(18)
    f.set_figheight(8)

    plt.rc('legend', fontsize=16)

    grade_averages_acc = [1 - avg for avg in grade_averages]
    combined_metric = (np.array(grade_averages_acc) + np.array(technique_unweighted_average_recalls)) / 2
    best_combined_avg_idx = np.argmax(combined_metric)

    plt.plot(x, technique_unweighted_average_recalls, label=f"UAR")
    plt.plot(x, grade_averages, label=f"UMAE")
    plt.plot(x, combined_metric, label=f"Combined score", color='black')

    plt.vlines(x=[best_combined_avg_idx], ymin=0, ymax=1, colors='black', ls='--', lw=2,
               label='Best combined score')

    plt.legend(loc='center right')
    plt.xlabel('Training steps')

    plt.xticks([x if i % 2 != 1 else '' for i, x in enumerate(x)])

    fig_save_path = os.path.join(eval_dir, f"knn_combined_{test_participant}")
    plt.savefig(fig_save_path)

    plt.rc('font', size=10)

    sns.set(font_scale=1.0)

    chosen_model_predictions_and_targets = predictions_and_targets_all[best_combined_avg_idx][0]

    technique_confusion_matrix_values = confusion_matrix(
        chosen_model_predictions_and_targets[1], chosen_model_predictions_and_targets[0]
    )

    df_cm = pd.DataFrame(technique_confusion_matrix_values,
                         index=[technique_idx_to_name_short[i] for i in technique_idx_to_name_short.keys()],
                         columns=[technique_idx_to_name_short[i] for i in technique_idx_to_name_short.keys()])

    plt.figure(figsize=(10, 7))
    s = sns.heatmap(df_cm, annot=True, cmap='Blues')
    s.set_xlabel('Predicted technique')
    s.set_ylabel('True technique')

    fig_save_path = os.path.join(eval_dir, f"knn_best_combined_technique_confusion_matrix_{test_participant}")
    plt.savefig(fig_save_path)

    chosen_model_predictions_and_targets = predictions_and_targets_all[best_combined_avg_idx][1]

    grade_confusion_matrix_values = confusion_matrix(
        chosen_model_predictions_and_targets[1], chosen_model_predictions_and_targets[0]
    )

    df_cm = pd.DataFrame(grade_confusion_matrix_values,
                         index=[grade_idx_to_name_short[i] for i in grade_idx_to_name_short.keys()],
                         columns=[grade_idx_to_name_short[i] for i in grade_idx_to_name_short.keys()])

    plt.figure(figsize=(10, 7))
    s = sns.heatmap(df_cm, annot=True, cmap='Blues')
    s.set_xlabel('Predicted grade')
    s.set_ylabel('True grade')
    fig_save_path = os.path.join(eval_dir, f"knn_best_combined_grade_confusion_matrix_{test_participant}")
    plt.savefig(fig_save_path)

    best_results = {
        "best technique checkpoint": str(checkpoints[best_technique_avg_idx]),
        "UAR of best technique checkpoint": str(technique_unweighted_average_recalls[best_technique_avg_idx]),
        "best grade checkpoint": str(checkpoints[best_grade_avg_idx]),
        "UMAE of best grade checkpoint": str(grade_averages[best_grade_avg_idx]),
        "best combined checkpoint": str(checkpoints[best_combined_avg_idx]),
        "UAR of best combined checkpoint": str(technique_unweighted_average_recalls[best_combined_avg_idx]),
        "UMAE of best combined checkpoint": str(grade_averages[best_combined_avg_idx]),
        "overall score of best combined checkpoint": str(combined_metric[best_combined_avg_idx])
    }

    best_results_save_path = os.path.join(eval_dir, f"knn_best_results_overview_{test_participant}.json")
    with open(best_results_save_path, 'w') as outfile:
        json.dump(best_results, outfile)


if __name__ == "__main__":
    main()
