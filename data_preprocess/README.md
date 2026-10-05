# SEED MATLAB reference utilities

These files were carried over from external preprocessing code and are retained
for reference. The current FACED and SEED Python pretraining paths do not call
`AutoICA_SEED.m`, `nt_interpolate_bad_channels_custom.m`, or the channel-name,
location, and coordinate files listed below. They are not required to run the
current Python pipeline. SEED `.mat` input files are read with SciPy, without a
MATLAB runtime.

The standalone MATLAB utility produces 58-channel `data_all_cleaned` and
`n_samples_one` output. The default SEED Python loader instead reads 62-channel,
200 Hz per-trial `*_eeg1` through `*_eeg15` arrays. This directory does not define
a required preprocessing stage or an alternative validated reproduction of the
paper's Python experiments.

`AutoICA_SEED.m` is a function rather than a machine-specific script. It takes
all input and output locations explicitly:

```matlab
AutoICA_SEED('/data/SEED/Preprocessed_EEG', ...
    '/data/SEED/chn_names.mat', ...
    '/data/SEED/SEED_10_20_standard.ced', ...
    '/data/SEED/SEED_coords_matrix.mat', ...
    '/data/SEED/processed');
```

Required toolboxes/files only if running this standalone MATLAB utility:

- FieldTrip;
- EEGLAB with ICLabel;
- NoiseTools and `nt_find_bad_channels_custom.m`;
- the original validated `nt_interpolate_bad_channels_custom.m` helper;
- `chn_names.mat`, a compatible EEGLAB channel-location `.ced` file, and
  `SEED_coords_matrix.mat` containing `coords_matrix`.

The custom interpolation helper is called inside this standalone utility but
is not currently included in this repository. Its absence does not block the
current Python pretraining, feature extraction, or MLP evaluation. Using this
utility as a separate experiment would require supplying its dependencies and
validating its output and downstream results.
