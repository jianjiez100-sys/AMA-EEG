"""Feature preprocessing used by the original FACED and SEED experiments."""
from pathlib import Path

import numpy as np

from .data_process import LDS


def seed_normalization_stats(data):
    # Preserve the source fast path: statistics are over last-axis time positions.
    means, variances, counts = [], [], []
    for subject in data:
        values = subject.reshape(-1, subject.shape[-1])
        means.append(values.mean(axis=0))
        variances.append(values.var(axis=0) * values.shape[0])
        counts.append(values.shape[0])
    return np.stack(means), np.stack(variances), np.asarray(counts)


def normalize_extraction_input(data, train_subs, dataset_name, seed_stats=None):
    """Fit on training subjects and apply the source dataset-specific transform."""
    train_subs = np.asarray(train_subs, dtype=int)
    if not train_subs.size:
        raise ValueError('At least one training subject is required')
    if dataset_name == 'FACED':
        train = data[train_subs]
        mean = np.mean(train, axis=(0, 1, 3), dtype=np.float32).reshape(-1, 1)
        var = np.var(train, axis=(0, 1, 3), dtype=np.float32)
        std = np.sqrt(var + 1e-5).reshape(-1, 1)
    elif dataset_name == 'SEED':
        means, variances, counts = seed_stats if seed_stats is not None else seed_normalization_stats(data)
        weights = counts[train_subs]
        mean = np.average(means[train_subs], axis=0, weights=weights)
        var = (np.average(variances[train_subs], axis=0, weights=weights) / weights.mean()
               + np.average((means[train_subs] - mean) ** 2, axis=0, weights=weights))
        std = np.sqrt(var) + 1e-5
        mean, std = mean.astype(np.float32), std.astype(np.float32)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    result = data.astype(np.float32)
    result -= mean
    result /= std
    return result


def smooth_trial_features(features, trial_counts):
    """Apply LDS independently to each subject/trial, never across boundaries."""
    counts = np.asarray(trial_counts, dtype=int).reshape(-1)
    if np.any(counts < 0) or counts.sum() != features.shape[1]:
        raise ValueError('Trial counts do not match the number of feature windows')
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    result = features.copy()
    for subject in range(len(result)):
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start < end:
                result[subject, start:end] = LDS(result[subject, start:end])
    return result


def load_split_seed_data(data_dir, time_len, time_step, n_subs):
    cache = Path(data_dir) / f'sliced_len{time_len}_step{time_step}_SEED'
    shards = cache / 'split_by_sub'
    if not shards.is_dir():
        return None
    files = sorted(shards.glob('*_data.npy'))
    if len(files) != n_subs:
        raise ValueError(f'Found {len(files)} SEED subject shards, expected {n_subs}')
    data = np.stack([np.load(path, mmap_mode='r') for path in files])
    metadata = cache if (cache / 'onesub_labels.npy').exists() else cache / 'metadata'
    return data, np.load(metadata / 'onesub_labels.npy'), np.load(metadata / 'n_samples_sessions.npy')
