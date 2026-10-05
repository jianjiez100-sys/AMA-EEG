# AMA-EEG: Adaptive Multimodal Alignment for Cross-Subject EEG Emotion Recognition

AMA-EEG aligns EEG representations with text and image semantics for
cross-subject emotion recognition. This repository supports the experiments
for FACED and SEED with a shared EEG backbone and separate pretraining protocols.
The source-code audit, restored defaults, and verification limits are recorded
in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Paper and citation

**AMA-EEG: Adaptive Multimodal Alignment for Cross-Subject EEG Emotion Recognition**<br>
Jianjie Zhou, Shilei Cao, Chengjian Xu, Zelin Liao, Haochuan Zhang, and Qingqing Zheng.

Published in *IEEE Transactions on Affective Computing*, 2026, pp. 1-10.

- [IEEE Xplore](https://ieeexplore.ieee.org/document/11668713)
- [DOI: 10.1109/TAFFC.2026.3728157](https://doi.org/10.1109/TAFFC.2026.3728157)

If you use AMA-EEG in your research, please cite:

```bibtex
@ARTICLE{11668713,
  author={Zhou, Jianjie and Cao, Shilei and Xu, Chengjian and Liao, Zelin and Zhang, Haochuan and Zheng, Qingqing},
  journal={IEEE Transactions on Affective Computing},
  title={AMA-EEG: Adaptive Multimodal Alignment for Cross-Subject EEG Emotion Recognition},
  year={2026},
  volume={},
  number={},
  pages={1-10},
  keywords={Electroencephalography;Modeling;Emotion recognition;Visualization;Faces;Training;Seeds (agriculture);Learning (artificial intelligence);Videos;Affective computing;Contrastive learning;EEG;emotion recognition;multimodal fusion;semantic alignment},
  doi={10.1109/TAFFC.2026.3728157}
}
```

Machine-readable citation metadata are also available in
[`CITATION.cff`](CITATION.cff).

## Supported tasks

| Config | Dataset | Classes | Validation |
| --- | --- | ---: | --- |
| `FACED` | FACED | 9 | 10-fold cross-subject |
| `FACED_def_c2` | FACED | 2 | 10-fold cross-subject |
| `SEED` | SEED | 3 | leave-one-subject-out |

The default model consumes EEG, text, and image features. Dataset-specific
pipeline details and validation records are available in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

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
SEED through its [official dataset page](https://bcmi.sjtu.edu.cn/home/seed/),
then place or link the processed data at the configured paths. SEED access is
limited to academic research, and downloaded dataset content must not be
redistributed; review the [SEED license
agreement](https://bcmi.sjtu.edu.cn/~seed/resource/license/SEED%20license.pdf)
before applying.

See [`DATA.md`](DATA.md) for the data/license boundary, local directory policy,
release-asset checksums, and the status of repository-hosted derived files.

```text
data/
├── FACED/                 # sub000.pkl, ..., sub122.pkl; 28 x 32 x 7500 at 250 Hz
└── SEED/                  # Official 200 Hz per-trial MAT files
    ├── 1/                # 15 subject files with *_eeg1, ..., *_eeg15
    ├── 2/
    └── 3/
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

Generate the offline projected arrays with the released `fusion1` weights:

```bash
python project_seed_features.py data=SEED
```

This applies the frozen alignment projector to each one-second feature before
the five-second average, preserving the source experiment. It saves
`multimodel_fusion/seed_fusion1/projected_text/` and `projected_image/`, which
SEED pretraining reads before applying its trainable residual projectors.
The command uses existing weights and does not retrain the alignment model.
FACED instead keeps its frozen `fusion10` alignment stage inside the model.

## Dynamic three-modal pretraining

Set a distinct `log.run` for each repeated experiment. Checkpoints are written
to `daest_cp/<dataset>/run<id>/`. FACED-9 and FACED-2 share the dataset name,
so give those tasks different run IDs as well.

FACED 9-class dynamic fusion:

```bash
python train_ext.py data=FACED train.pretrain_mode=2 log.run=1
```

FACED binary dynamic fusion:

```bash
python train_ext.py data=FACED_def_c2 train.pretrain_mode=2 log.run=2
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

### Minimal smoke test

This command runs one SEED fold for one epoch with one training batch and one
validation batch. After generating SEED's projected features, it verifies data loading, forward/backward,
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
python extract_features.py data=FACED_def_c2 log.run=2
python train_mlp.py data=FACED_def_c2 log.run=2

python extract_features.py data=SEED log.run=1
python train_mlp.py data=SEED log.run=1
```

Extracted EEG features are saved under
`extracted_features/<dataset>/run<id>/`. They are derived intermediate results,
so users can regenerate them from the released code and their checkpoints.
`extract_features.py` fits input normalization on training subjects, extracts
1024-dimensional backbone features in float32, and applies LDS independently
within each subject/trial. Both original extraction scripts disable running
normalization. Its full code block is retained, commented out, immediately before
LDS in `extract_features.py`: FACED reorders videos to playback order and restores
the standard order afterward; SEED handles sessions using variable trial counts.
Input `normTrain` is separate from this disabled feature step. FACED extraction
reads subject PKL files; SEED prefers per-subject NPY shards and otherwise reads
the packed cache or session MAT files. FACED tiles one-subject labels, while SEED
keeps full packed labels when provided. SEED's original fast normalization operates along time positions;
see the audit for this source-code caveat. The MLP fits StandardScaler on
training features and clips transformed values to [-3, 3].

## Paper results and reproducibility

The paper reports the following fold-level mean and standard
deviation. Accuracy, macro F1, and Cohen's kappa are reported in percent.

| Task | Validation | Accuracy | F1 | Kappa |
| --- | --- | ---: | ---: | ---: |
| FACED-2 | 10-fold cross-subject | 78.29 ± 3.43 | 78.45 ± 3.32 | 56.57 ± 6.85 |
| FACED-9 | 10-fold cross-subject | 61.30 ± 6.79 | 61.42 ± 6.86 | 56.38 ± 7.67 |
| SEED-3 | leave-one-subject-out | 69.45 ± 10.87 | 66.21 ± 13.84 | 54.19 ± 16.20 |

These are historical paper results, not results re-established by the source
parity repair. The current local source configurations differ between datasets:

| Setting | FACED | SEED |
| --- | --- | --- |
| Seed; window/stride | 7; 5 s / 2 s | 7; 5 s / 2 s |
| Pretraining optimizer; lr; weight decay | Adam; 7e-4; 1.5e-4 | Adam; 7e-4; 1.5e-4 |
| Maximum/minimum epochs; patience | 15 / 3; 3 | 25 / 10; 5 |
| Samples per subject pair/session | 1 | 10 |
| Effective paired batch | 56 | 30 |
| Probe loss weight; fusion temperature | 2; 0.1 | 1; 0.05 |
| MLP lr; maximum epochs | 2e-4; 30 | 5e-4; 20 |

Both MLPs use Adam, weight decay `2.2e-3`, batch size 256, hidden dimensions
[512, 128], dropout 0.2, and patience 10. FACED binary pretraining samples all
28 videos and removes the neutral class before forward propagation; the effective
batch becomes 48. Its downstream labels follow the loader's actual order:
negative 0-3, neutral 4, positive 5-8. The source binary scripts contain conflicting
label constants, so the historical binary result still requires independent
verification. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Qwen2-VL-7B-Instruct caption generation is a one-time offline preprocessing
step. The paper reports approximately 61 minutes for FACED and 75
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
are in `cfgs/data/`. `cfgs/protocol/` selects each source experiment's training
defaults automatically with `data`. Any setting can be overridden from the command line:

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

The current FACED and SEED Python pretraining paths do not call MATLAB or
perform bad-channel interpolation. FACED reads the processed `.pkl` input;
SEED reads 62-channel, 200 Hz per-trial `.mat` files using `scipy.io.loadmat`,
resamples to 125 Hz, normalizes, and slices in Python. Existing compatible
slice caches are loaded directly. Reading a `.mat` file does not require MATLAB.

`data_preprocess/AutoICA_SEED.m` and
`data_preprocess/nt_find_bad_channels_custom.m` were carried over from external
preprocessing code and are retained as reference utilities. They are not
prerequisites for the current Python experiment. Only running `AutoICA_SEED.m`
separately requires FieldTrip, EEGLAB with ICLabel, NoiseTools, the custom
interpolation helper, and channel-name/location/coordinate files; see
[`data_preprocess/README.md`](data_preprocess/README.md). This standalone utility
outputs 58 channels and a different MAT structure, which is not the input to the
default SEED loader. For an additional preprocessing reference, see
[EEG_Preprocess_python_new](https://github.com/soul-M-42/EEG_Preprocess_python_new).

## Contributing, security, and license

- Contribution and local validation instructions: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Private security or sensitive-data reports: [`SECURITY.md`](SECURITY.md)
- Code license: [`LICENSE`](LICENSE) (MIT)
- Dataset and derived-file terms: [`DATA.md`](DATA.md)

## Contact

For questions about the paper or this repository, contact Jianjie Zhou at
[SUAT25060448@stu.suat-sz.edu.cn](mailto:SUAT25060448@stu.suat-sz.edu.cn).
