# Source protocol audit and reproducibility

Audit date: 2026-10-05. The reference experiments are the current local copies
of `DAEST_Multimodel_new_fusion_recheck` (FACED) and
`DAEST_SEED_Multiomdel_projector1` (SEED). This repair restores their data and
training computations. These checks do not establish that every current source
default matches an archived paper run. Historical results in README remain
unverified by this repair.

## What differed in the release

The previous unified release changed SEED's input format, normalized EEG using
the FACED rule, moved SEED's frozen alignment projection inside the model after
window averaging, applied FACED's probes/fusion and false-negative mask to SEED,
reduced SEED sampling from 10 repeats to 1, and reused FACED epoch/MLP settings.
Feature extraction also bypassed both source scripts' training-subject input
normalization and trial-wise LDS, and used mixed-precision inference.
These changes affect the learned model and classifier inputs.

The restored release shares the EEG backbone while selecting separate
Lightning modules, losses, and protocol defaults. Relative paths, fold ranges,
and run-specific output directories remain configurable.

## Pipeline comparison

| Stage | FACED current release | SEED current release |
| --- | --- | --- |
| EEG input | Subject `.pkl`, 28 videos x 32 channels x 7500 points, 250 Hz | Three session directories, 15 subjects/session, 15 `*_eegN` trials, 62 channels, 200 Hz |
| Resampling | SciPy Fourier resample to 125 Hz | SciPy Fourier resample to 125 Hz |
| Initial normalization | Subject-wide scalar mean/std, calculated after excluding absolute values above 30 times median absolute amplitude | Separate mean/std for every trial/channel, epsilon 1e-8 |
| Windows | 5 s, stride 2 s: 625 points, 250-point step, 13 windows per 30-second video | 5 s, stride 2 s; variable number per video, truncated to the minimum across subjects within each session/video |
| Semantic input | Raw 1024-D CLIP text/image window arrays | Offline fusion1-projected `(windows, 5, 1024)` text/image arrays |
| Frozen alignment | fusion10 inside the model; window average precedes alignment | fusion1 outside the model; project each second before window average |
| Trainable projection | Residual EEG/text/image projectors | Residual EEG/text/image projectors |
| Probe input | Raw CLIP features by default | Normalized offline aligned teacher features, before online projectors |
| Probe loss multiplier | 2 | 1 |
| Negative mask | Same video, different subject | Same video/time content ID regardless of subject or session; diagonal remains positive |
| Sampling | One sample per video per subject pair, 56 examples | Same-session subject pairs, 10 repeats, 30 examples; 2730 steps/epoch with 14 training subjects and three sessions |
| Epoch max/min; stopping patience | 15/3; 3 in current source config | 25/10; 5 |
| Pretraining optimizer | Adam, lr 7e-4, weight decay 1.5e-4, cosine warm restarts, InfoNCE temperature 0.07 | Same |
| Extraction input normalization | Training-subject channel statistics; sqrt(var + 1e-5) | Source fast path uses last-axis time-position statistics; sqrt(var) + 1e-5 |
| Extracted representation | 1024-D backbone before the EEG projector; stratified normalization disabled; float32 inference | Same |
| Temporal feature processing | LDS separately per subject/video; running normalization disabled | LDS separately per subject/session/video; running normalization disabled |
| Classifier | Train-only StandardScaler, clip [-3,3], MLP 1024 -> 512 -> 128 -> 9 (or 2), dropout 0.2, no BN | Same, output dimension 3 |
| MLP optimizer/epochs | Adam lr 2e-4, max/min 30/10, weight decay 0.0022, patience 10 | Adam lr 5e-4, max/min 20/10, same decay/patience |

Both pipelines use seed 7. FACED default pretraining uses float32 matmul setting
`medium`; SEED uses `high`, matching their training entry points. Extraction is
float32 (`32-true`) and preserves each source script's matmul setting.

## Window alignment and the commented feature step

FACED calculates one window count from its fixed 30-second clips: at 5 s / 2 s,
each of its 28 videos has 13 windows. SEED calculates counts per subject,
session, and video, then takes `min(counts, axis=0)` over subjects. Each
session/video keeps its own count, so videos of different lengths still have
different window counts. The existing SEED EEG cache has counts ranging from
91 to 131 per video and 5016 windows per subject. Multimodal pretraining applies
a further per-video minimum over EEG, text, and image lengths; pure EEG feature
extraction uses the complete EEG cache.

After backbone inference the original intended sequence is feature running
normalization followed by LDS. Both source scripts comment out running
normalization and keep LDS enabled. `extract_features.py` now retains the full
running-normalization implementation as commented code at that exact position.
It includes training-feature mean/variance, FACED playback-order reordering
and restoration, and SEED session boundaries calculated from variable trial
counts. It can be uncommented to run that separate ablation. The enabled EEG
input `normTrain` occurs before inference and is distinct from feature running
normalization.

FACED extraction loads raw PKL and saves a one-subject label template. SEED
prefers float32 subject NPY shards with metadata, otherwise loads packed NPY
or cold session MAT; its labels may already span all subjects. The inference
loader explicitly preserves those label formats. LDS uses the per-session,
per-video counts, never a fixed number of windows for SEED.
FACED's retained playback-order path reads the released MATLAB 5 remark files
with SciPy; this avoids an incompatibility between hdf5storage 0.1.19 and
NumPy 2 when uncommenting running normalization.

## Source caveats and explicit release choices

- **SEED extraction normalization:** the source comment calls the fast path
  channel normalization, but `reshape(-1, shape[-1])` operates on the 625 time
  positions. The restored default preserves the executable code. Changing this
  to channel statistics would be a new experiment, not a parity repair.
- **FACED binary labels:** the loader's nine-class order is negative 0-3,
  neutral 4, positive 5-8. The source pretraining filter follows that order,
  but its downstream binary constants use neutral 8 and reversed groups. Its
  24-video loader path also conflicts with hardcoded 28-video sampling/mapping.
  The release uses a 28-video nine-label cache, removes label 4 in pretraining,
  and maps labels >=5 to positive consistently in the MLP. Nine-class numerical
  parity is verified; exact historical binary parity is not established.
- **Optional DE baseline:** the source direct-DE branch uses the number of
  points per window as the filtering sampling rate. This branch is preserved
  and was not part of the verified default learned-feature pipeline.
- **MATLAB reference utility:** `AutoICA_SEED.m` was carried over from external
  preprocessing code. Neither the current FACED nor SEED Python pretraining
  path calls it, its interpolation helper, or its channel auxiliary files.
  SEED reads MAT files through SciPy; MATLAB is not required. The standalone
  utility's 58-channel `data_all_cleaned` output is not the 62-channel per-trial
  input used by the default Python SEED experiment.
- **Historical configs:** FACED archived run configs and the current editable
  YAML have different stopping/temperature settings. This repair adopts the
  current source config; reproducing a specific published run requires that
  run's config, checkpoint, and complete fold outputs.

Old caches produced by the previous SEED merged-MAT loader should be replaced
with a fresh cache under the official session-directory input. For FACED binary,
use a 28-video cache with labels 0-8; the dataset rejects a 24-video cache.
Changed protocols require new pretraining checkpoints and extracted features.

## Verification performed

Environment: Python 3.10, PyTorch 2.5.1+cu121, Lightning 2.6.5,
NumPy 2.2.6, SciPy 1.15.3. Numerical checks and pipeline smoke tests ran on CPU,
with seed 7 and four CPU threads; no full dataset training was performed.

1. `tools/check_source_parity.py` loads definitions from both reference projects
   as numerical oracles. With identical weights/input/dropout seeds, restored
   FACED and SEED match exactly for dynamic training loss and all 44 parameter
   tensors receiving gradients, backbone predictions, MLP outputs, paired sampler
   sequences, fold input normalization, and trial LDS. FACED common validation
   metrics also match. The protocol tests check SEED validation separately.
   FACED fixture loss is 12.85471249; SEED fixture loss is 5.58000898; maximum
   gradient difference is 0 for both. These are fixture losses, not EEG accuracy.
2. All 15 text and 15 image files regenerated from the release archives and
   saved fusion1 weights match the source's stored projected arrays within
   floating-point tolerance. Text max absolute difference: 3.3378601e-6,
   RMSE 1.5994126e-7. Image max absolute difference: 3.9339066e-6,
   RMSE 2.2819045e-7. The released projector checksums also match the source.
   Read-only checks of existing real EEG caches also compare the first and last
   window of every trial for the first/last subject: all 112 FACED and 180 SEED
   sample tuples match the corresponding source dataset exactly, including EEG,
   labels, semantic inputs, and IDs. The aligned subject lengths are 364 for
   FACED and 5001 for SEED; SEED extraction still uses the full 5016 EEG windows
   per subject before teacher-length truncation.
3. FACED-9, SEED-3, and FACED-2 each complete one CPU pretraining epoch with one
   training/validation batch, checkpoint loading, extraction with normalization
   and LDS, and one MLP epoch. FACED fixtures use four subjects, two folds,
   5-second windows with a 25-second stride to keep the check small; SEED uses
   three subjects, one session, and 5-second windows with a 2-second stride.
   Finite extracted shapes are FACED `(224,1024)`, SEED `(90,1024)`, and FACED-2
   `(224,1024)` before neutral filtering. The binary MLP uses 96 training and
   96 validation windows after filtering. Fixture accuracy is not a scientific
   reproduction result. The SEED three-stage smoke test also passes with fresh
   run 9931 and finite `(90,1024)` features.
4. Fifteen unit checks cover training and validation fusion, configuration
   selection, cold SEED loading/resampling,
   channel/trial normalization, shortest-subject truncation, multimodal index
   mapping, content/event IDs, 2730-step sampling, train-only fold statistics,
   LDS boundaries, variable video/session lengths, FACED PKL versus SEED shard
   loading and labels, FACED playback-order round trips, rejection of incompatible
   old SEED caches, and repository checks. Syntax and Hydra config checks also
   pass.
5. The retained running-normalization block was temporarily uncommented in a
   verification harness. FACED (including playback reorder/restoration) and
   SEED (three sessions with variable trial counts) match their source functions
   exactly. Re-extracting the FACED-9, SEED-3, and FACED-2 smoke checkpoints with
   the commented block disabled produces arrays identical to those saved before
   this follow-up change.

Local verification logs and fixtures are under `/tmp/ama_source_parity_smoke/`,
`/tmp/ama_projected_asset_parity.log`, and `/tmp/ama_real_cached_samples.log`;
these paths are not release assets.
Follow-up slicing/extraction checks are recorded in
`/tmp/ama_slice_extraction_tests.log` and `/tmp/ama_extraction_update_verify.log`.
Follow-up model checks are recorded in `/tmp/ama_public_fusion_tests.log` and
`/tmp/ama_public_fusion_source_checks.log`.
The updated SEED smoke log is `/tmp/ama_seed_public_ce_smoke.log`, with
stage outputs under `/tmp/ama_seed_public_ce_smoke/`.

Run portable checks:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
python train_ext.py data=FACED --cfg job --resolve
python train_ext.py data=FACED_def_c2 --cfg job --resolve
python train_ext.py data=SEED --cfg job --resolve
git diff --check
```

If the source projects are available locally, run the additional numerical
comparison (the projects are only read):

```bash
python tools/check_source_parity.py \
  --faced-source /path/to/DAEST_Multimodel_new_fusion_recheck \
  --seed-source /path/to/DAEST_SEED_Multiomdel_projector1
```

## Reference snapshot

SHA-256 of the audited source files, so later source edits can be distinguished
from changes in this release:

| Project | File | SHA-256 |
| --- | --- | --- |
| FACED | `cfgs/config.yaml` | `8dc51d6120e0aa77993eea106ce40d864d3fde0d7c501ede9b70c3ce0fc41f34` |
| FACED | `data/io_utils.py` | `84280961af4c6d78907ebbbc95235fc800b0a184fc1487205ac8ab387b7d4160` |
| FACED | `model/pl_models.py` | `fba5a491dcc788bd0d30a72c99a04fb31be16a09dc1633a4a93cd3cc39afbdc2` |
| FACED | `ext_fea_reorder.py` | `437eaa9520a299e09472e700141dcde18d41fc170745ae023552b94c9ba52560` |
| SEED | `cfgs/config.yaml` | `d4f53c0686a9ef40cfb4e6e27bdb8bb37e767ddb9a3933bbf5f1e05308440486` |
| SEED | `data/io_utils.py` | `425bad589a14b78c78b6fe580f36f2fe4a771ba03967c8d6c049ff3c304c670d` |
| SEED | `model/pl_models.py` | `f68e853f5247e184fa5d53495266de345245a97c2994a3312e9449ca0d90f0c4` |
| SEED | `extract_fea.py` | `afe35bb9752390563efac8be903ae5faec365f351d7164f46c2b25ea905a384a` |
