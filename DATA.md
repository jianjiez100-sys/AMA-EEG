# Data and derived-asset policy

This repository contains source code, small configuration files, selected
derived stimulus features, and trained alignment projectors. It does not grant
permission to redistribute the underlying FACED or SEED EEG recordings.

## Raw EEG datasets

- **FACED:** request access from the [official Synapse
  page](https://www.synapse.org/Synapse:syn50614194/wiki/620378).
- **SEED:** request access from the dataset's official provider.

Place the datasets locally under `data/FACED/` and `data/SEED/`, or override
`data.data_dir` on the command line. These directories are ignored by Git.
Users are responsible for complying with the datasets' access agreements,
ethics approvals, and restrictions on redistribution.

## Released text and image features

The v1.1.0 release assets contain stimulus-derived CLIP features, not raw EEG.
They are required for the multimodal pretraining stage. Verify downloads with:

```bash
sha256sum <downloaded-file>
```

| Release asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `faced_image_features_clip_vit_centercrop_timelen5_timestep2.tar.gz` | 1,371,467 | `eb1652d3858f4584b8e0bc43561992e83798b93cc40b63cad02fb183cef37a3a` |
| `faced_text_features_timelen5_timestep2_1024.tar.gz` | 2,065,546 | `c2ec75b830667e2597a286a00fa66b8c6bcb0f93c07532e3a968a0a90070ecc2` |
| `seed_image_features_clip_vit_centercrop_timelen5_timestep2.tar.gz` | 13,318,538 | `14729b66d5ba8c030588216b86a95f5552505cd10c445940ae81de6a5edb8d16` |
| `seed_text_features_timelen5_timestep2_1024.tar.gz` | 13,512,754 | `c465dd4a04745f859192826722727fdc871f7c4acc9e966b8feec4f686545cc2` |

The source stimuli and pretrained models retain their own terms. The repository
MIT License does not replace those terms.

## Alignment projectors and analysis arrays

The `.pt` files under `multimodel_fusion/fusion10/` and
`multimodel_fusion/seed_fusion1/` are trained text/image alignment projectors.
The `fusion10/projected_text/` and `fusion10/projected_image/` arrays are
stimulus-derived analysis artifacts used to inspect the learned alignment;
they are not raw or participant-level EEG and are not required for training.
See the README in each alignment directory for provenance and checksums.

## Participant-level behavioral files

`After_remarks/` contains 123 coded-subject MATLAB files. Each file stores 28
records with the fields `score`, `trial`, `vid`, `Accuracy`, and
`ResponseTime`. These are participant-level derived behavioral records, not
software. The project maintainer has confirmed that they may be publicly
redistributed in this repository. They are **not covered by the MIT code
license**; downstream reuse remains subject to the applicable participant
consent, ethics approval, and source-dataset terms.

## Generated local outputs

The following paths are intentionally ignored because they can be regenerated
or may contain machine-specific information:

- `features/`, `extracted_features/`, `pretrained_projectors/`
- `daest_cp/`, `checkpoints_mlp/`, `wandb/`, `logs/`, `outputs/`
- `data/FACED/`, `data/SEED/`

Do not commit API keys, access tokens, raw participant identifiers, local
absolute paths, or logs containing those values.
