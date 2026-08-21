# FACED feature preprocessing

Install the optional environment before running the scripts:

```bash
python -m pip install -r requirements-preprocess.txt
```

Generate objective Qwen descriptions with `FACED_Qwen_nolabel.py`, then use
the text/image feature scripts to create the sliding-window semantic features.
The scripts use repository-relative defaults; review the model ID, frame path,
output path, and cache path near the top of each script before a full run.

The EEG windowing utility uses explicit command-line paths:

```bash
python FACED/move_window_timelen5_timestep2_EEG.py \
  --input-file /path/to/raw_eeg_features.npy \
  --output-dir /path/to/processed_features
```

Run `python FACED/move_window_timelen5_timestep2_EEG.py --help` to view shape
and sliding-window overrides.
