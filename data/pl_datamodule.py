from .dataset import EEG_Dataset, FACED_Dataset_new, SEED_Dataset_new, SEEDV_Dataset_new, PretrainSampler
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import torch

class EEGDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_subs, val_subs,
                 train_vids, val_vids, loo=False, num_workers=8):
        super().__init__()
        self.load_dir = os.path.join(cfg.data_dir,'processed_data')
        self.save_dir = os.path.join(cfg.data_dir,'sliced_data')

        self.num_workers = num_workers
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.loo = loo
        self.train_vids = train_vids
        self.val_vids = val_vids

        self.cfg = cfg

        self.sliced_data_dir = os.path.join(self.save_dir, f'sliced_len{cfg.timeLen}_step{cfg.timeStep}')

    def prepare_data(self):
        EEG_Dataset(self.cfg,sliced=False)
        print('prepare data finished!')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = EEG_Dataset(self.cfg,train_subs=self.train_subs,mods='train',sliced=True)
            self.valset = EEG_Dataset(self.cfg,val_subs=self.val_subs,mods='val',sliced=True)
        if stage == 'validate':
            self.valset = EEG_Dataset(self.cfg,val_subs=self.val_subs,mods='val',sliced=True)

    def train_dataloader(self):
        self.n_samples_sessions = np.load(self.sliced_data_dir+'/metadata/n_samples_sessions.npy')
        train_sampler = PretrainSampler(n_subs=len(self.train_subs), batch_size=len(self.train_vids),
                                            n_samples_session=self.n_samples_sessions, n_times=1)
        return DataLoader(self.trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=self.num_workers)


    def val_dataloader(self):
        self.n_samples_sessions = np.load(self.sliced_data_dir+'/metadata/n_samples_sessions.npy')
        val_sampler = PretrainSampler(len(self.val_subs), batch_size=len(self.val_vids),
                                    n_samples_session=self.n_samples_sessions, n_times=1, if_val_loo=self.loo)
        return DataLoader(self.valset, batch_sampler=val_sampler, pin_memory=True, num_workers=self.num_workers)


class SEEDVDataModule(pl.LightningDataModule):
    def __init__(self, load_dir, save_dir, timeLen, timeStep, train_subs, val_subs, train_vids, val_vids,
                 n_session=3, fs=125, n_chans=60, n_subs=16, n_vids = 15, n_class=5, loo=False, num_workers=8):
        super().__init__()
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.num_workers = num_workers
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.loo = loo
        self.train_vids = train_vids
        self.val_vids = val_vids
        self.n_session = n_session
        self.fs = fs
        self.n_chans = n_chans
        self.n_subs = n_subs
        self.n_vids = n_vids
        self.n_class = n_class

        self.sliced_data_dir = os.path.join(self.save_dir, f'sliced_len{self.timeLen}_step{self.timeStep}')

    def prepare_data(self):
        SEEDV_Dataset_new(self.load_dir,self.save_dir,self.timeLen,self.timeStep,sliced=False,
                          n_session=self.n_session, fs=self.fs, n_chans=self.n_chans,
                          n_subs=self.n_subs, n_vids = self.n_vids, n_class=self.n_class)
        print('prepare data finished!')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = SEEDV_Dataset_new(self.load_dir,self.save_dir,self.timeLen,self.timeStep,
                                              train_subs=self.train_subs,mods='train',sliced=True,
                                              n_session=self.n_session, fs=self.fs, n_chans=self.n_chans,
                                              n_subs=self.n_subs, n_vids = self.n_vids, n_class=self.n_class)
            self.valset = SEEDV_Dataset_new(self.load_dir,self.save_dir,self.timeLen,self.timeStep,
                                            val_subs=self.val_subs,mods='val',sliced=True,
                                            n_session=self.n_session, fs=self.fs, n_chans=self.n_chans,
                                            n_subs=self.n_subs, n_vids = self.n_vids, n_class=self.n_class)
        if stage == 'validate':
            self.valset = SEEDV_Dataset_new(self.load_dir,self.save_dir,self.timeLen,self.timeStep,
                                            val_subs=self.val_subs,mods='val',sliced=True,
                                            n_session=self.n_session, fs=self.fs, n_chans=self.n_chans,
                                            n_subs=self.n_subs, n_vids = self.n_vids, n_class=self.n_class)

    def train_dataloader(self):
        self.n_samples_sessions = np.load(self.sliced_data_dir+'/metadata/n_samples_sessions.npy')
        train_sampler = PretrainSampler(n_subs=len(self.train_subs), batch_size=len(self.train_vids),
                                            n_samples_session=self.n_samples_sessions, n_times=1)
        return DataLoader(self.trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        self.n_samples_sessions = np.load(self.sliced_data_dir+'/metadata/n_samples_sessions.npy')
        val_sampler = PretrainSampler(len(self.val_subs), batch_size=len(self.val_vids),
                                        n_samples_session=self.n_samples_sessions, n_times=1,if_val_loo=self.loo)
        return DataLoader(self.valset, batch_sampler=val_sampler, pin_memory=True, num_workers=self.num_workers)


class FACEDDataModule(pl.LightningDataModule):
    def __init__(self, load_dir, save_dir, timeLen, timeStep, train_subs, val_subs, train_vids, val_vids,
                 n_session=1, fs=125, n_chans=30, n_subs=123, n_vids=28, n_class=9, loo=False, num_workers=8,
                 image_feat_dir=None, text_feat_dir=None, use_pretrain_sampler=False):
        super().__init__()
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.num_workers = num_workers
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.loo = loo
        self.train_vids = train_vids
        self.val_vids = val_vids
        self.n_session = n_session
        self.fs = fs
        self.n_chans = n_chans
        self.n_subs = n_subs
        self.n_vids = n_vids
        self.n_class = n_class
        self.image_feat_dir = image_feat_dir
        self.text_feat_dir = text_feat_dir
        self.use_pretrain_sampler = use_pretrain_sampler
        self.sliced_data_dir = os.path.join(self.save_dir, f'sliced_len{self.timeLen}_step{self.timeStep}')

    def prepare_data(self):
        FACED_Dataset_new(self.load_dir, self.save_dir, self.timeLen, self.timeStep, sliced=False,
                          n_session=self.n_session, fs=self.fs, n_chans=self.n_chans,
                          n_subs=self.n_subs, n_vids=self.n_vids, n_class=self.n_class)
        print('prepare data finished!')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = FACED_Dataset_new(self.load_dir, self.save_dir, self.timeLen, self.timeStep,
                                              train_subs=self.train_subs, mods='train', sliced=True,
                                              n_session=self.n_session, fs=self.fs, n_chans=self.n_chans,
                                              n_subs=self.n_subs, n_vids=self.n_vids, n_class=self.n_class,
                                              image_feat_dir=self.image_feat_dir,
                                              text_feat_dir=self.text_feat_dir)

            self.valset = FACED_Dataset_new(self.load_dir, self.save_dir, self.timeLen, self.timeStep,
                                            val_subs=self.val_subs, mods='val', sliced=True,
                                            n_session=self.n_session, fs=self.fs, n_chans=self.n_chans,
                                            n_subs=self.n_subs, n_vids=self.n_vids, n_class=self.n_class,
                                            image_feat_dir=self.image_feat_dir,
                                            text_feat_dir=self.text_feat_dir)

        if stage == 'validate':
            self.valset = FACED_Dataset_new(self.load_dir, self.save_dir, self.timeLen, self.timeStep,
                                            val_subs=self.val_subs, mods='val', sliced=True,
                                            n_session=self.n_session, fs=self.fs, n_chans=self.n_chans,
                                            n_vids=self.n_vids, n_class=self.n_class,
                                            image_feat_dir=self.image_feat_dir, text_feat_dir=self.text_feat_dir)

    def train_dataloader(self):
        if self.use_pretrain_sampler:
            n_samples_sessions = np.load(
                os.path.join(self.sliced_data_dir, 'metadata', 'n_samples_sessions.npy'))
            train_sampler = PretrainSampler(
                n_subs=len(self.train_subs), batch_size=len(self.train_vids),
                n_samples_session=n_samples_sessions, n_times=1)
            return DataLoader(self.trainset, batch_sampler=train_sampler,
                            pin_memory=True, num_workers=self.num_workers)
        else:
            return DataLoader(self.trainset, batch_size=256, shuffle=True,
                            pin_memory=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        if self.use_pretrain_sampler:
            n_samples_sessions = np.load(
                os.path.join(self.sliced_data_dir, 'metadata', 'n_samples_sessions.npy'))
            val_sampler = PretrainSampler(
                len(self.val_subs), batch_size=len(self.val_vids),
                n_samples_session=n_samples_sessions, n_times=1, if_val_loo=self.loo)
            return DataLoader(self.valset, batch_sampler=val_sampler,
                            pin_memory=True, num_workers=self.num_workers)
        else:
            return DataLoader(self.valset, batch_size=256, shuffle=False,
                            pin_memory=True, num_workers=self.num_workers, drop_last=False)


class SEEDDataModule(pl.LightningDataModule):
    def __init__(self, load_dir, save_dir, timeLen, timeStep, train_subs, val_subs,
                 train_vids, val_vids, n_session=3, fs=125, n_chans=62, n_subs=15,
                 n_vids=15, n_class=3, loo=True, num_workers=8, image_feat_dir=None,
                 text_feat_dir=None, sampler_times=1, cross_session=False):
        super().__init__()
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.train_subs = list(train_subs)
        self.val_subs = list(val_subs)
        self.train_vids = train_vids
        self.val_vids = val_vids
        self.n_session = n_session
        self.fs = fs
        self.n_chans = n_chans
        self.n_subs = n_subs
        self.n_vids = n_vids
        self.n_class = n_class
        self.loo = loo
        self.num_workers = num_workers
        self.image_feat_dir = image_feat_dir
        self.text_feat_dir = text_feat_dir
        self.sampler_times = sampler_times
        self.cross_session = cross_session
        self.sliced_data_dir = os.path.join(save_dir, f'sliced_len{timeLen}_step{timeStep}_SEED')

    def prepare_data(self):
        SEED_Dataset_new(
            self.load_dir, self.save_dir, self.timeLen, self.timeStep, sliced=False,
            n_session=self.n_session, fs=self.fs, n_chans=self.n_chans,
            n_subs=self.n_subs, n_vids=self.n_vids, n_class=self.n_class)

    def setup(self, stage=None):
        if stage in ('fit', None):
            common = dict(
                load_dir=self.load_dir, save_dir=self.save_dir, timeLen=self.timeLen,
                timeStep=self.timeStep, sliced=True, n_session=self.n_session, fs=self.fs,
                n_chans=self.n_chans, n_subs=self.n_subs, n_vids=self.n_vids,
                n_class=self.n_class, image_feat_dir=self.image_feat_dir,
                text_feat_dir=self.text_feat_dir)
            self.trainset = SEED_Dataset_new(
                train_subs=self.train_subs, mods='train', **common)
            self.valset = SEED_Dataset_new(
                val_subs=self.val_subs, mods='val', **common)

    def train_dataloader(self):
        n_samples_sessions = self.trainset.n_samples_aligned.reshape(
            self.n_session, self.n_vids)
        sampler = PretrainSampler(
            n_subs=len(self.train_subs), batch_size=len(self.train_vids),
            n_samples_session=n_samples_sessions, n_times=self.sampler_times,
            cross_session=self.cross_session)
        return DataLoader(
            self.trainset, batch_sampler=sampler, pin_memory=True,
            persistent_workers=self.num_workers > 0, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=256, shuffle=False, pin_memory=True,
            persistent_workers=self.num_workers > 0, num_workers=self.num_workers)

if __name__ == '__main__':
    pass
