import glob
import os
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, TensorDataset

from data.feature_processing import (
    load_split_seed_data, normalize_extraction_input,
    seed_normalization_stats, smooth_trial_features,
)
from data.io_utils import load_finetune_EEG_data
from model import get_extractor_class


def load_extraction_data(cfg):
    """Read FACED raw PKL or SEED subject shards/session MAT with trial metadata."""
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    data_dir = to_absolute_path(data_cfg.data_dir)
    if data_cfg.dataset_name == 'SEED':
        split = load_split_seed_data(data_dir, data_cfg.timeLen2, data_cfg.timeStep2, data_cfg.n_subs)
        if split is not None:
            return split
    elif data_cfg.dataset_name == 'FACED':
        # The source extracts all 28 videos before downstream binary filtering.
        data_cfg.n_class, data_cfg.n_vids = 9, 28
        if (Path(data_dir) / 'all_pkl_files').is_dir():
            data_dir = str(Path(data_dir) / 'all_pkl_files')
        if len(list(Path(data_dir).glob('*.pkl'))) != data_cfg.n_subs:
            raise ValueError(f'Expected {data_cfg.n_subs} FACED subject .pkl files in {data_dir}')
    data, labels, _, trial_counts = load_finetune_EEG_data(data_dir, data_cfg)
    data = np.asarray(data, dtype=np.float32)
    data = data.reshape(data_cfg.n_subs, -1, data.shape[-2], data.shape[-1])
    return data, labels, trial_counts


def extraction_labels(labels, n_subs, n_windows, dataset_name):
    """FACED stores one subject's labels; SEED may store the full packed array."""
    labels = np.asarray(labels).reshape(-1)
    if dataset_name == 'SEED' and len(labels) == n_subs * n_windows:
        return labels
    if len(labels) != n_windows:
        raise ValueError(
            f'{dataset_name} extraction labels have length {len(labels)}, '
            f'expected {n_windows} per subject or {n_subs * n_windows} total for SEED')
    return np.tile(labels, n_subs)


def direct_de_features(data, trial_counts):
    import mne

    n_subs, n_windows, n_channels, points = data.shape
    bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]
    features = np.zeros((n_subs, n_windows, n_channels, len(bands)))
    boundaries = np.concatenate(([0], np.cumsum(np.asarray(trial_counts).reshape(-1))))
    for band_index, (low, high) in enumerate(bands):
        for subject in range(n_subs):
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if start >= end:
                    continue
                trial = data[subject, start:end].transpose(1, 0, 2).reshape(n_channels, -1)
                # Preserve the source DE branch's use of window points as sfreq.
                filtered = mne.filter.filter_data(
                    trial.astype(np.float64), points, low, high, verbose=False)
                filtered = filtered.reshape(n_channels, -1, points)
                features[subject, start:end, :, band_index] = (
                    0.5 * np.log(2 * np.pi * np.e * np.var(filtered, axis=2))).T
    return features.reshape(n_subs, n_windows, -1)


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def extract_features(cfg: DictConfig) -> None:
    """Apply the source fold normalization, backbone inference, and trial LDS."""
    torch.set_float32_matmul_precision(cfg.ext_fea.matmul_precision)
    output_dir = to_absolute_path(cfg.ext_fea.output_dir)
    checkpoint_dir = os.path.join(
        to_absolute_path(cfg.log.cp_dir), cfg.data.dataset_name, f"run{cfg.log.run}")
    os.makedirs(output_dir, exist_ok=True)
    data, labels, trial_counts = load_extraction_data(cfg)
    np.save(os.path.join(output_dir, 'onesub_label2.npy'), labels)

    n_folds = cfg.data.n_subs if cfg.train.valid_method == 'loo' else int(cfg.train.valid_method)
    end_fold = n_folds if cfg.get('end_fold') is None else min(int(cfg.end_fold), n_folds)
    folds = list(range(int(cfg.get('start_fold', 0)), end_fold))
    if cfg.train.iftest:
        folds = folds[:1]
    n_per = round(cfg.data.n_subs / n_folds)
    seed_stats = (seed_normalization_stats(data)
                  if cfg.ext_fea.normTrain and cfg.data.dataset_name == 'SEED' else None)

    with open_dict(cfg.model):
        cfg.model.proj_type = 'residual'
        cfg.model.use_ln_backbone = (
            not cfg.train.use_original_sampling if cfg.data.dataset_name == 'SEED' else True)
    with open_dict(cfg.train):
        for modality in ('text', 'image'):
            key = f'pretrained_{modality}_proj'
            if cfg.train[key]:
                cfg.train[key] = to_absolute_path(cfg.train[key])

    for fold in folds:
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)
        train_subs = sorted(set(range(cfg.data.n_subs)) - set(val_subs))
        fold_data = (normalize_extraction_input(data, train_subs, cfg.data.dataset_name, seed_stats)
                     if cfg.ext_fea.normTrain else data)
        if cfg.ext_fea.use_pretrain:
            matches = sorted(glob.glob(os.path.join(checkpoint_dir, f'fold{fold}_*.ckpt')))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f'Expected one checkpoint for fold {fold} in {checkpoint_dir}, found {len(matches)}')
            extractor = get_extractor_class(cfg.data.dataset_name)(
                hydra.utils.instantiate(cfg.model), cfg.train)
            checkpoint = torch.load(matches[0], map_location='cpu', weights_only=False)
            state = checkpoint.get('state_dict', checkpoint)
            state = {key: value for key, value in state.items()
                     if not key.startswith(('text_probe.', 'image_probe.'))}
            missing, unexpected = extractor.load_state_dict(state, strict=False)
            invalid_missing = [key for key in missing
                               if not key.startswith(('text_probe.', 'image_probe.'))]
            if invalid_missing or unexpected:
                raise ValueError(f'Checkpoint mismatch: missing={invalid_missing}, unexpected={unexpected}')
            extractor.model.set_stratified([])
            extractor.eval()
            extractor.freeze()
            flat_data = fold_data.reshape(-1, fold_data.shape[-2], fold_data.shape[-1])
            fold_labels = extraction_labels(
                labels, cfg.data.n_subs, data.shape[1], cfg.data.dataset_name)
            dataset = TensorDataset(torch.from_numpy(flat_data).unsqueeze(1), torch.from_numpy(fold_labels))
            loader = DataLoader(dataset, batch_size=cfg.ext_fea.batch_size, shuffle=False,
                                num_workers=cfg.train.num_workers)
            trainer = pl.Trainer(
                logger=False, accelerator=cfg.train.accelerator,
                devices=1 if cfg.train.accelerator == 'cpu' else cfg.train.gpus,
                precision=cfg.ext_fea.precision, enable_checkpointing=False)
            predictions = trainer.predict(extractor, loader)
            features = torch.cat(predictions, dim=0).cpu().numpy()
            features = features.reshape(cfg.data.n_subs, -1, features.shape[-1])
        else:
            features = direct_de_features(fold_data, trial_counts)

        # Step 1: Running normalization of EXTRACTED FEATURES (disabled).
        # Both original scripts comment out this block for the current ablation.
        # The input normTrain above is a separate, enabled preprocessing step.
        # To reproduce the alternative with running norm, uncomment this block
        # before LDS. FACED needs chronological video reorder and restoration;
        # SEED already follows session/trial order and uses variable trial lengths.
        # from data.data_process import running_norm_onesubsession
        # train_features = features[train_subs]
        # data_mean = np.mean(np.mean(train_features, axis=1), axis=0)
        # data_var = np.mean(np.var(train_features, axis=1), axis=0)
        #
        # if cfg.data.dataset_name == 'FACED':
        #     from utils.reorder_vids import (
        #         video_order_load, reorder_vids_sepVideo, reorder_vids_back,
        #     )
        #     # Extraction includes all 28 videos, even for binary classification.
        #     n_vids = np.asarray(trial_counts).shape[1]
        #     vid_order = video_order_load(n_vids)
        #     features, vid_play_order = reorder_vids_sepVideo(
        #         features, vid_order, np.arange(n_vids), n_vids)
        #
        # session_counts = np.sum(trial_counts, axis=1)
        # session_boundaries = np.concatenate(([0], np.cumsum(session_counts)))
        # for subject in range(cfg.data.n_subs):
        #     for start, end in zip(session_boundaries[:-1], session_boundaries[1:]):
        #         features[subject, start:end] = running_norm_onesubsession(
        #             features[subject, start:end], data_mean, data_var,
        #             cfg.ext_fea.rn_decay)
        #
        # if cfg.data.dataset_name == 'FACED':
        #     features = reorder_vids_back(features, n_vids, vid_play_order)

        # Step 2: LDS smoothing (enabled in both original scripts).
        # trial_counts is FACED's equal 28-video counts, or SEED's variable
        # session/video counts; each subject/trial is smoothed independently.
        if cfg.ext_fea.lds:
            features = smooth_trial_features(features, trial_counts)
        features = features.reshape(-1, features.shape[-1])
        filename = f"{cfg.data.dataset_name.lower()}_run{cfg.log.run}_f{fold}_fea_{cfg.ext_fea.mode}.npy"
        np.save(os.path.join(output_dir, filename), features)
        print(f"Saved fold {fold}: {features.shape} -> {os.path.join(output_dir, filename)}")


if __name__ == '__main__':
    extract_features()
