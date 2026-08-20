import glob
import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from data.dataset import FACED_Dataset_new, SEED_Dataset_new
from model.pl_models import ExtractorModel


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def extract_features(cfg: DictConfig) -> None:
    """Extract raw backbone features from each pretrained fold."""
    pl.seed_everything(cfg.seed)
    data_dir = to_absolute_path(cfg.data.data_dir)
    output_dir = to_absolute_path(cfg.ext_fea.output_dir)
    checkpoint_dir = os.path.join(
        to_absolute_path(cfg.log.cp_dir), cfg.data.dataset_name, f"run{cfg.log.run}")
    os.makedirs(output_dir, exist_ok=True)

    n_folds = cfg.data.n_subs if cfg.train.valid_method == 'loo' else int(cfg.train.valid_method)
    end_fold = n_folds if cfg.get('end_fold') is None else min(int(cfg.end_fold), n_folds)
    folds = list(range(int(cfg.get('start_fold', 0)), end_fold))
    if cfg.train.iftest:
        folds = folds[:1]

    common = dict(
        load_dir=data_dir, save_dir=data_dir, timeLen=cfg.data.timeLen,
        timeStep=cfg.data.timeStep, train_subs=list(range(cfg.data.n_subs)),
        mods='train', sliced=True, n_session=cfg.data.n_session, fs=cfg.data.fs,
        n_chans=cfg.data.n_channs, n_subs=cfg.data.n_subs, n_vids=cfg.data.n_vids,
        n_class=cfg.data.n_class)
    if cfg.data.dataset_name == 'FACED':
        dataset = FACED_Dataset_new(**common)
    elif cfg.data.dataset_name == 'SEED':
        dataset = SEED_Dataset_new(**common)
    else:
        raise ValueError(f"Feature extraction is not configured for {cfg.data.dataset_name}")

    np.save(os.path.join(output_dir, 'onesub_label2.npy'), dataset.onesub_labels.numpy())
    loader = DataLoader(
        dataset, batch_size=cfg.ext_fea.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True)

    with open_dict(cfg.model):
        cfg.model.proj_type = 'residual'
        cfg.model.use_ln_backbone = True
    with open_dict(cfg.train):
        if cfg.train.pretrained_text_proj:
            cfg.train.pretrained_text_proj = to_absolute_path(cfg.train.pretrained_text_proj)
        if cfg.train.pretrained_image_proj:
            cfg.train.pretrained_image_proj = to_absolute_path(cfg.train.pretrained_image_proj)

    for fold in folds:
        matches = sorted(glob.glob(os.path.join(checkpoint_dir, f'fold{fold}_*.ckpt')))
        if not matches:
            raise FileNotFoundError(f"No checkpoint found for fold {fold} in {checkpoint_dir}")
        base_model = hydra.utils.instantiate(cfg.model)
        extractor = ExtractorModel.load_from_checkpoint(
            matches[-1], model=base_model, cfg=cfg.train, strict=True)
        extractor.model.set_stratified([])
        extractor.eval()
        extractor.freeze()
        trainer = pl.Trainer(
            logger=False, accelerator=cfg.train.accelerator,
            devices=1 if cfg.train.accelerator == 'cpu' else cfg.train.gpus,
            precision=cfg.train.precision, enable_checkpointing=False)
        predictions = trainer.predict(extractor, loader)
        features = torch.cat(predictions, dim=0).cpu().numpy()
        filename = f"{cfg.data.dataset_name.lower()}_run{cfg.log.run}_f{fold}_fea_{cfg.ext_fea.mode}.npy"
        np.save(os.path.join(output_dir, filename), features)
        print(f"Saved fold {fold}: {features.shape} -> {os.path.join(output_dir, filename)}")


if __name__ == '__main__':
    extract_features()
