# AMA-EEG: Adaptive Multimodal Alignment for Cross-Subject EEG Emotion Recognition

AMA-EEG aligns EEG representations with text and image semantics for
cross-subject emotion recognition. This repository supports the experiments
for FACED and SEED with one shared model and dataset-specific configurations.

## Supported tasks

| Config | Dataset | Classes | Validation |
| --- | --- | ---: | --- |
| `FACED` | FACED | 9 | 10-fold cross-subject |
| `FACED_def_c2` | FACED | 2 | 10-fold cross-subject |
| `SEED` | SEED | 3 | leave-one-subject-out |

The default model consumes three modalities: EEG, text features, and image
features. In dynamic fusion mode, class probes estimate the confidence of the
text and image modalities and use it to update their fusion weights.

## Installation

Python 3.10 and an NVIDIA GPU are recommended. The tested environment uses
PyTorch 2.5.1, CUDA 12.1, and PyTorch Lightning 2.6.5.

```bash
git clone git@github.com:jianjiez100-sys/AMA-EEG.git
cd AMA-EEG
conda create -n ama-eeg python=3.10 -y
conda activate ama-eeg
pip install -r requirements.txt
```

CPU execution is also available by adding
`train.accelerator=cpu train.precision=32-true` to a command.

## Data and features

Raw EEG data are not redistributed by this repository. Download FACED from
[Synapse](https://www.synapse.org/Synapse:syn50614194/wiki/620378) and request
SEED from its official provider, then place or link the processed data at the
configured paths:

```text
data/
├── FACED/
└── SEED/
```

The default paths can be overridden without editing YAML, for example:

```bash
python train_ext.py data=SEED data.data_dir=/path/to/SEED_EEG_data
```

### FACED text and image features

FACED raw CLIP text and image features are published in
[GitHub Release v1.1.0](https://github.com/jianjiez100-sys/AMA-EEG/releases/tag/v1.1.0):

- [Text features](https://github.com/jianjiez100-sys/AMA-EEG/releases/download/v1.1.0/faced_text_features_timelen5_timestep2_1024.tar.gz)
- [Image features](https://github.com/jianjiez100-sys/AMA-EEG/releases/download/v1.1.0/faced_image_features_clip_vit_centercrop_timelen5_timestep2.tar.gz)

Extract both archives under `features/` so that the directory names match
`cfgs/data/FACED.yaml`:

```text
features/
├── text_timelen5_timestep2_1024_objective/
└── image_features_clip_vit_centercrop_timelen5_timestep2/
```

Each directory contains 28 sliding-window feature files with 1024-dimensional
CLIP features. The scripts under `FACED/` can be used to regenerate them.

### SEED text and image features

SEED raw 1024-dimensional text and image features are also distributed as
GitHub Release assets instead of regular Git files:

- [SEED text features](https://github.com/jianjiez100-sys/AMA-EEG/releases/download/v1.1.0/seed_text_features_timelen5_timestep2_1024.tar.gz)
- [SEED image features](https://github.com/jianjiez100-sys/AMA-EEG/releases/download/v1.1.0/seed_image_features_clip_vit_centercrop_timelen5_timestep2.tar.gz)

Extract both archives under `features/SEED/`:

```text
features/SEED/
├── text_timelen5_timestep2_1024/
│   ├── negative/
│   ├── neutral/
│   └── positive/
└── image_features_clip_vit_centercrop_timelen5_timestep2/
    ├── negative/
    ├── neutral/
    └── positive/
```

The SEED configuration uses the paper's `fusion1` text/image alignment weights
from `multimodel_fusion/seed_fusion1/`. The model loads and freezes these
weights, then trains its online residual projectors. Only `fusion1` is included;
the unused fusion2/fusion3/fusion4 experiments and precomputed
`projected_text/` or `projected_image/` arrays are not part of the code release.

## Dynamic three-modal pretraining

Set a distinct `log.run` for each repeated experiment. Checkpoints are written
to `daest_cp/<dataset>/run<id>/`.

FACED 9-class dynamic fusion:

```bash
python train_ext.py data=FACED train.pretrain_mode=2 log.run=1
```

FACED binary dynamic fusion:

```bash
python train_ext.py data=FACED_def_c2 train.pretrain_mode=2 log.run=1
```

SEED 3-class dynamic fusion:

```bash
python train_ext.py data=SEED train.pretrain_mode=2 log.run=1
```

The available pretraining modes are:

| `train.pretrain_mode` | Alignment target |
| ---: | --- |
| `0` | EEG + text |
| `1` | EEG + image |
| `2` | EEG + dynamically weighted text/image fusion |
| `3` | EEG + static text/image fusion |

For mode 3, set the text weight with `train.fusion_alpha`; the image weight is
`1 - train.fusion_alpha`.

### Minimal smoke test

This command runs one SEED fold for one epoch with one training batch and one
validation batch. It verifies data loading, fusion1 loading, forward/backward,
dynamic weights, and checkpoint writing:

```bash
python train_ext.py \
  data=SEED \
  train.pretrain_mode=2 \
  train.iftest=true \
  train.max_epochs=1 \
  train.min_epochs=1 \
  train.num_workers=0 \
  train.limit_train_batches=1 \
  train.limit_val_batches=1 \
  start_fold=0 end_fold=1 \
  log.run=999
```

## Downstream classification

After pretraining, extract the EEG backbone feature for every fold. Use the
same dataset config and `log.run` as pretraining:

```bash
python extract_features.py data=FACED log.run=1
python train_mlp.py data=FACED log.run=1
```

Equivalent binary and SEED pipelines are:

```bash
python extract_features.py data=FACED_def_c2 log.run=1
python train_mlp.py data=FACED_def_c2 log.run=1

python extract_features.py data=SEED log.run=1
python train_mlp.py data=SEED log.run=1
```

Extracted EEG features are saved under
`extracted_features/<dataset>/run<id>/`. They are derived intermediate results,
so users can regenerate them from the released code and their checkpoints.
`extract_features.py` exports raw backbone features; it does not apply the
running normalization or LDS smoothing used by older experiment scripts.

## Configuration

The main settings are in `cfgs/config.yaml`; dataset paths and class definitions
are in `cfgs/data/`. Any setting can be overridden from the command line:

```bash
python train_ext.py \
  data=FACED \
  data.data_dir=/path/to/FACED \
  train.gpus='[1]' \
  train.max_epochs=20 \
  log.run=2
```

Use `python train_ext.py --cfg job data=SEED` to inspect the resolved job
configuration without starting training.

## Preprocessing

`data_preprocess/AutoICA_SEED.m` and
`data_preprocess/nt_find_bad_channels_custom.m` provide MATLAB preprocessing
utilities for SEED. `AutoICA_SEED.m` requires
[FieldTrip](https://www.fieldtriptoolbox.org/). For an additional preprocessing
reference, see
[EEG_Preprocess_python_new](https://github.com/soul-M-42/EEG_Preprocess_python_new).
