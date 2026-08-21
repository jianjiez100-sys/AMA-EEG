# Contributing to AMA-EEG

Thank you for helping improve AMA-EEG. Small, focused changes with a clear
reproduction or maintenance benefit are easiest to review.

## Development setup

```bash
git clone https://github.com/jianjiez100-sys/AMA-EEG.git
cd AMA-EEG
conda create -n ama-eeg python=3.10 -y
conda activate ama-eeg
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install `requirements-preprocess.txt` only when changing the optional visual or
text feature-generation scripts.

## Before opening a pull request

Run the checks that do not require restricted datasets:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
python train_ext.py --cfg job data=FACED
python train_ext.py --cfg job data=SEED
git diff --check
```

If a change affects training, also report the dataset/config, fold range,
random seed, hardware, relevant command, and before/after metrics. Do not attach
raw FACED/SEED data or participant-level files to an issue or pull request.

## Pull requests

- Create a focused branch such as `fix/config-validation`.
- Explain the problem, the change, and how it was tested.
- Keep paths configurable and platform-independent.
- Update README/config documentation when behavior changes.
- Do not commit generated checkpoints, feature directories, credentials, or
  local environment files.

By contributing code, you agree that your contribution can be distributed
under the repository's MIT License. This statement does not cover third-party
datasets, model weights, or manuscripts.
