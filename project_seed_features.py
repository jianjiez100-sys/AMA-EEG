"""Generate the source SEED offline features using the released fusion1 weights."""
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import nn

from data.dataset import SEED_FILE_MAPPING
from model.models import ResidualAdd


def project_directory(input_dir, output_dir, weight_path):
    projector = nn.Sequential(
        nn.Linear(1024, 1024),
        ResidualAdd(nn.Sequential(nn.GELU(), nn.Linear(1024, 1024), nn.Dropout(0.2))),
        nn.LayerNorm(1024))
    state = torch.load(weight_path, map_location='cpu', weights_only=True)
    projector.load_state_dict({key.removeprefix('net.'): value for key, value in state.items()})
    projector.eval()
    with torch.no_grad():
        for relative_path in SEED_FILE_MAPPING:
            source = Path(input_dir) / relative_path
            raw = np.load(source).astype(np.float32)
            if raw.ndim != 3 or raw.shape[1:] != (5, 1024):
                raise ValueError(f'Expected (windows, 5, 1024), got {raw.shape}: {source}')
            # Projection precedes the five-second average, as in SEED_fusion/fusion1.
            projected = projector(torch.from_numpy(raw.reshape(-1, 1024))).numpy().reshape(raw.shape)
            destination = Path(output_dir) / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, projected)
            print(f'Saved {destination}: {projected.shape}')


@hydra.main(config_path='cfgs', config_name='config', version_base='1.3')
def project_seed_features(cfg: DictConfig) -> None:
    if cfg.data.dataset_name != 'SEED':
        raise ValueError('Run this command with data=SEED')
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    for modality in ('text', 'image'):
        project_directory(
            to_absolute_path(cfg.data[f'raw_{modality}_feat_dir']),
            to_absolute_path(cfg.data[f'{modality}_feat_dir']),
            to_absolute_path(cfg.data[f'pretrained_{modality}_proj']))


if __name__ == '__main__':
    project_seed_features()
