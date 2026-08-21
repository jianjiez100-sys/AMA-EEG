# SEED MATLAB preprocessing

`AutoICA_SEED.m` is a function rather than a machine-specific script. It takes
all input and output locations explicitly:

```matlab
AutoICA_SEED('/data/SEED/Preprocessed_EEG', ...
    '/data/SEED/chn_names.mat', ...
    '/data/SEED/SEED_10_20_standard.ced', ...
    '/data/SEED/SEED_coords_matrix.mat', ...
    '/data/SEED/processed');
```

Required MATLAB toolboxes/files:

- FieldTrip;
- EEGLAB with ICLabel;
- NoiseTools and `nt_find_bad_channels_custom.m`;
- the original validated `nt_interpolate_bad_channels_custom.m` helper;
- `chn_names.mat`, a compatible EEGLAB channel-location `.ced` file, and
  `SEED_coords_matrix.mat` containing `coords_matrix`.

The custom interpolation helper is required to reproduce the original
pipeline but is not currently included in this repository. Do not substitute a
different interpolation algorithm when reproducing the paper without
revalidating the preprocessing and downstream results.
