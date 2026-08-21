# AMA-EEG: Adaptive Multimodal Alignment for Cross-Subject EEG Emotion Recognition

AMA-EEG aligns EEG representations with text and image semantics for
cross-subject emotion recognition. This repository supports the experiments
for FACED and SEED with one shared model and dataset-specific configurations.

## Paper and citation

**AMA-EEG: Adaptive Multimodal Alignment for Cross-Subject EEG Emotion Recognition**<br>
Jianjie Zhou, Shilei Cao, Chengjian Xu, Zelin Liao, Haochuan Zhang, and Qingqing Zheng.

Accepted for publication in *IEEE Transactions on Affective Computing*;
the DOI and IEEE Xplore link are forthcoming. The accepted manuscript is not
redistributed in this repository. After publication, this section and
[`CITATION.cff`](CITATION.cff) will be updated with the version-of-record link.

If you use this repository before the final bibliographic record is available,
please cite the accepted paper title above and the software metadata in
[`CITATION.cff`](CITATION.cff).

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
git clone https://github.com/jianjiez100-sys/AMA-EEG.git
cd AMA-EEG
conda create -n ama-eeg python=3.10 -y
conda activate ama-eeg
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To regenerate the visual and textual semantic features with the scripts under
`FACED/`, install the optional preprocessing dependencies as well:

```bash
python -m pip install -r requirements-preprocess.txt
```

Verify the installation before downloading data:

```bash
python -c "import torch, pytorch_lightning, torchmetrics; print(torch.__version__, torch.cuda.is_available())"
python train_ext.py --cfg job data=SEED
```

CPU execution is also available by adding
`train.accelerator=cpu train.precision=32-true` to a command.

## Data and features

Raw EEG data are not redistributed by this repository. Download FACED from
[Synapse](https://www.synapse.org/Synapse:syn50614194/wiki/620378) and request
SEED from its official provider, then place or link the processed data at the
configured paths:

See [`DATA.md`](DATA.md) for the data/license boundary, local directory policy,
release-asset checksums, and the status of repository-hosted derived files.

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

## Paper results and reproducibility

The accepted paper reports the following fold-level mean and standard
deviation. Accuracy, macro F1, and Cohen's kappa are reported in percent.

| Task | Validation | Accuracy | F1 | Kappa |
| --- | --- | ---: | ---: | ---: |
| FACED-2 | 10-fold cross-subject | 78.29 ± 3.43 | 78.45 ± 3.32 | 56.57 ± 6.85 |
| FACED-9 | 10-fold cross-subject | 61.30 ± 6.79 | 61.42 ± 6.86 | 56.38 ± 7.67 |
| SEED-3 | leave-one-subject-out | 69.45 ± 10.87 | 66.21 ± 13.84 | 54.19 ± 16.20 |

The paper configuration uses seed 7, 5-second windows with a 2-second stride,
a maximum of 15 alignment epochs, AdamW with learning rate `7e-4` and weight
decay `1.5e-4`, and early stopping with patience 5. The paired-subject sampler
produces effective batches of 56 for FACED (2 × 28 videos) and 30 for SEED
(2 × 15 videos). The downstream MLP uses a maximum of 30 epochs, Adam with
learning rate `2e-4`, weight decay `2.2e-3`, batch size 256, dropout 0.2, and
early-stopping patience 10.

Qwen2-VL-7B-Instruct caption generation is a one-time offline preprocessing
step. The accepted paper reports approximately 61 minutes for FACED and 75
minutes for SEED on one NVIDIA GeForce RTX 4090. End-to-end training time and
memory use depend on the selected task and machine; record them together with
the resolved Hydra configuration when reporting a new run.

Use the same `data`, `log.run`, and fold range in all three stages. A paper
reproduction consists of:

```text
train_ext.py -> extract_features.py -> train_mlp.py -> fold mean ± standard deviation
```

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
[FieldTrip](https://www.fieldtriptoolbox.org/), EEGLAB with ICLabel, the
NoiseTools bad-channel utilities, a channel-location file, and a coordinate
matrix. Call the function with explicit paths as documented in its header; it
does not use machine-specific paths. For an additional preprocessing reference,
see
[EEG_Preprocess_python_new](https://github.com/soul-M-42/EEG_Preprocess_python_new).

## Contributing, security, and license

- Contribution and local validation instructions: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Private security or sensitive-data reports: [`SECURITY.md`](SECURITY.md)
- Code license: [`LICENSE`](LICENSE) (MIT)
- Dataset and derived-file terms: [`DATA.md`](DATA.md)
