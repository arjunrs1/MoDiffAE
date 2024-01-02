# MoDiffAE - Motion Diffusion Autoencoder

<p float="left">
  <img src="images/technique_mod/high_kick.gif" width="250" />
  <img src="images/technique_mod/technique_arrow.png" width="250" /> 
  <img src="images/technique_mod/low_kick.gif" width="250" />
</p>

<p float="left">
  <img src="images/skill_level_mod/low_skill.gif" width="250" />
  <img src="images/skill_level_mod/skill_arrow.png" width="250" /> 
  <img src="images/skill_level_mod/high_skill.gif" width="250" />
</p>

This is the official repository of MoDiffAE, the first sequence-to-sequence model capable of modifying arbitrary human motion characteristics. Below you will find an overview of the MoDiffAE architecture. It is based on Diffusion Models and Transformers. Although its capabilities are demonstrated on human motion, the model could in theory be used for the modification of any type of sequence data. However, it has not been tested on other types of sequence data, yet.


![image](images/architecture_overview.svg)

![image](images/modiffae_architecture_detailed.svg)

## Setutp
Setup and activate conda environment:
```bash
conda env create -f environment.yml
conda activate modiffae
```

## Preprocessing 

Before training, the data was preprocessed in multiple steps. An overview of the pipeline is shown below. All these steps as well as all results are completely reproducable. You can download the raw as well as the preprocessed data from my [cloud](https://e.pcloud.link/publink/show?code=kZFusjZ5d1c0YIA6Xp0gEYKxQdzdFIJSGT7). If you do not want to change the preprocessing, I would recommend the preprocessed data set. If you want to reproduce or adjust the preprocessing, use the raw data set and execute the preprocessing pipeline by running the following command: 

```bash
bash preprocessing/karate/preprocess.sh
```
This will automatically perform all preprocessing steps and create splits for training, validation and testing.

![image](images/preprocessing_overview.png)

## Training

### Core 
```bash
python -m training.train --model_type modiffae --save_dir {path_to_save_dir} --test_participant {test_participant} --pose_rep {pose_representation}
```

### Regressor 
```bash
python -m training.train --model_type semantic_regressor --modiffae_model_path {path_to_modiffae_core_model}
```

### Generator (experimental)
```bash
python -m training.train --model_type semantic_generator --modiffae_model_path {path_to_modiffae_core_model}
```

## Evaluation

### Core 
```bash
python -m evaluation.semantic_embedding_knn_regression --save_dir {path_to_save_dir}
```
```bash
python -m evaluation.visualize_semantic_embedding --modiffae_model_path {path_to_modiffae_core_model}
```

### Regressor 
```bash
python -m evaluation.semantic_regressor_eval --modiffae_model_path {path_to_modiffae_core_model} --save_dir {path_to_save_dir}
```

### Generator (experimental)
```bash
python -m evaluation.generation_fid --modiffae_model_path {path_to_modiffae_core_model}
```

## Data generation (experimental)
```bash
python -m sample.rejection_generation --modiffae_model_path {path_to_modiffae_core_model} --semantic_generator_model_path {path_to_generator_model} --semantic_regressor_model_path {path_to_regressor_model}
```
