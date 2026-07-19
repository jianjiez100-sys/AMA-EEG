<!-- omit in toc -->
# DAEST

<!-- omit in toc -->
## Table of Contents
- [Main Article](#main-article)
- [Provided Datasets](#provided-datasets)
- [Data Preprocessing](#data-preprocessing)
- [Requirements](#requirements)
  - [Mandatory](#mandatory)
  - [Optional](#optional)
  - [Hardware](#hardware)
- [Getting Started](#getting-started)

## Main Article

<details>
  <summary>Dynamic-Attention-based EEG State Transition Modeling for Emotion Recognition</summary>
  Electroencephalogram (EEG)-based emotion decoding can objectively quantify people's emotional state and has broad application prospects in human-computer interaction and early detection of emotional disorders. Recently emerging deep learning architectures have significantly improved the performance of EEG emotion decoding. However, existing methods still fall short of fully capturing the complex spatiotemporal dynamics of neural signals, which are crucial for representing emotion processing. This study proposes a Dynamic-Attention-based EEG State Transition (DAEST) modeling method to characterize EEG spatiotemporal dynamics. The model extracts spatiotemporal components of EEG that represent multiple parallel neural processes and estimates dynamic attention weights on these components to capture transitions in brain states. The model is optimized within a contrastive learning framework for cross-subject emotion recognition. The proposed method achieved state-of-the-art performance on three publicly available datasets: FACED, SEED, and SEED-V. It achieved $75.4\pm5.5\%$ accuracy in the binary classification of positive and negative emotions and $59.3\pm7.7\%$ in nine-class discrete emotion classification on the FACED dataset, $88.1\pm3.6\%$ in the three-class classification of positive, negative, and neutral emotions on the SEED dataset, and $73.6\pm12.7\%$ in five-class discrete emotion classification on the SEED-V dataset. The learned EEG spatiotemporal patterns and dynamic transition properties offer valuable insights into neural dynamics underlying emotion processing.

  ![DAEST](paper_img/DAEST.jpg)


</details>

## Provided Datasets

- [SEED/SEEDV](https://bcmi.sjtu.edu.cn/~seed/seed-v.html)
- [FACED](https://www.synapse.org/Synapse:syn50614194/wiki/620378)

## Data Preprocessing
For **SEED dataset preprocessing**, we provide two MATLAB scripts in `data_preprocess/` folder:

- `nt_find_bad_channels_custom.m`  
  Custom bad-channel detection function.

- `AutoICA_SEED.m`  
  Automatic ICA-based preprocessing for SEED EEG data.

The second script requires the **[FieldTrip toolbox](https://www.fieldtriptoolbox.org/)** installed in your MATLAB environment.

For preprocessing of other datasets, you may also refer to this repository:  
👉 [EEG_Preprocess_python_new](https://github.com/soul-M-42/EEG_Preprocess_python_new)


## Requirements

### Mandatory

| Package          | Minimum | Recommended |
|------------------|---------|-------------|
| python           | 3.8.0   | 3.12.0      |
| numpy            | 1.21.0  | 1.23.0      |
| hydra-core       | 1.3.0   | 1.3.0       |
| omegaconf        | 2.3.0   | 2.3.0       |
| torch            | 2.0.0   | 2.5.0       |
| pytorch-lightning| 2.0.0   | 2.4.0       |
| wandb            | 0.18.0  | 0.18.6      |
| mne              | 1.8.0   | 1.8.0       |
| h5py             | 3.3.0   | 3.12.1      |
| hdf5storage      | 0.1.19  | 0.1.19      |


### Optional

| Package     | Minimum | Recommended |
|-------------|---------|-------------|
| CUDA        | 11.6    | 12.1        |
| transformers| 4.21.0  | 4.46.2      |

### Hardware

- **Recommended Operating System:** Linux
- **Recommended GPU:** NVIDIA RTX 4090 (any GPU/CPU works)

## Getting Started

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/jianjiez100-sys/DAEST.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd DAEST
    ```

3. **Create your conda environment and install dependencies:**

    ```bash
    conda create -n DAEST python=3.12.0
    pip install torch>=2.0 pytorch-lightning>=2.0 hydra-core==1.3 wandb mne h5py hdf5storage joblib numpy omegaconf
    ```

    Note: No `requirements.txt` is checked into the repo — create one from the package list above.

4. **Download pre-extracted features:**

    We provide pre-extracted text and image features (CLIP ViT-B, 1024-dim, sliding-window sliced).
    Download from [GitHub Releases]() and extract to `./features/`:

    ```
    features/
    ├── text_timelen5_timestep2_1024_objective/      # 28 .npy files, ~7.3 MB
    └── image_features_clip_vit_centercrop_timelen5_timestep2/  # 28 .npy files, ~1.6 MB
    ```

    These features are derived from publicly available pretrained models (CLIP)
    and contain no raw video or EEG data. To extract them yourself, see
    scripts in the `FACED/` directory.

5. **Configure paths and settings:**

    🔧 Files you may need to edit (most paths are now relative, but verify them):
    - `cfgs/data/FACED.yaml` — set your EEG dataset directory (FACED requires [Synapse registration](https://www.synapse.org/Synapse:syn50614194/wiki/620378))
    - `train_ext.py` — feature and checkpoint directories (default: `./features/` and `./daest_cp/`)
    - `train_mlp.py` — feature path for finetuning (default: `./extracted_features/`)

    See `cfgs/readme_cfg.md` for additional config guidance.

6. **Run pretraining:**

    ```bash
    python train_ext.py log.run=1 data=FACED model=cnn_att train.gpus=[0] train.valid_method=loo
    ```

7. **Run finetuning:**

    ```bash
    python train_mlp.py log.run=1 data=FACED mlp.wd=0.0022
    ```

    **Important:** Use the same `log.run` number across both stages within a trial.

8. **Use wandb to see results.**

9. **Note on the full pipeline:** The feature extraction stage (`ext_fea_reorder.py`) that sits between pretraining and finetuning (running norm + LDS smoothing + saving `.npy` features) is not included in this checkout. This checkout directly loads pre-extracted features for finetuning.
