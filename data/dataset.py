from torch.utils.data import Dataset, DataLoader, Sampler
import torch
# 确保这里能正确导入你的 io_utils
from .io_utils import (
    load_EEG_data,
    load_processed_FACED_NEW_data,
    load_processed_SEED_NEW_data,
    load_processed_SEEDV_NEW_data,
    save_sliced_data,
)
import os
import numpy as np
import random
from functools import partial


# ==========================================
# 核心修改：FACED_Dataset_new
# ==========================================
class FACED_Dataset_new(Dataset):
    def __init__(self, load_dir, save_dir, timeLen, timeStep, train_subs=None, val_subs=None, sliced=True, mods='train',
                 n_session=1, fs=125, n_chans=30, n_subs=123, n_vids=28, n_class=9,
                 image_feat_dir=None, text_feat_dir=None):

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.n_subs = n_subs
        self.n_vids = n_vids
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.mods = mods

        # [新增] 保存多模态路径
        self.image_feat_dir = image_feat_dir
        self.text_feat_dir = text_feat_dir
        self.has_multimodal = False

        self.sliced_data_dir = os.path.join(self.save_dir, f'sliced_len{self.timeLen}_step{self.timeStep}')
        self.load_processed_data = partial(load_processed_FACED_NEW_data, dir=self.load_dir,
                                           timeLen=self.timeLen, timeStep=self.timeStep)
        self.save_sliced_data = partial(save_sliced_data, sliced_data_dir=self.sliced_data_dir)

        if not sliced:
            if not os.path.exists(self.sliced_data_dir + '/saved.npy'):
                print('slicing processed dataset')
                data, onesub_labels, n_samples_onesub, n_samples_sessions = self.load_processed_data(
                    fs=fs, n_chans=n_chans, n_session=n_session, n_subs=n_subs, n_vids=n_vids, n_class=n_class)
                self.save_sliced_data(data=data, onesub_labels=onesub_labels, n_samples_onesub=n_samples_onesub,
                                      n_samples_sessions=n_samples_sessions)
            else:
                print('sliced data exist!')

        self.onesub_labels = torch.from_numpy(
            np.load(os.path.join(self.sliced_data_dir, 'metadata', 'onesub_labels.npy')))
        self.labels = self.onesub_labels.repeat(self.n_subs)
        self.onesubLen = len(self.onesub_labels)

        n_samples_onesub_arr = np.load(os.path.join(self.sliced_data_dir, 'metadata', 'n_samples_onesub.npy'))
        self.samples_per_vid = int(n_samples_onesub_arr[0])

        # ================= [关键修改] 构建文件名映射列表 =================
        # FACED 顺序: Anger(3), Disgust(3), Fear(3), Sadness(3), Neutral(4), Amusement(3), Inspiration(3), Joy(3), Tenderness(3)
        # 总共 28 个视频，对应 vid 0 ~ 27
        negative = [
            f"neg_{emotion}_{i}_features.npy"
            for emotion in ('a', 'd', 'f', 's') for i in range(1, 4)
        ]
        neutral = [f"neu_{i}_features.npy" for i in range(1, 5)]
        positive = [
            f"pos_{emotion}_{i}_features.npy"
            for emotion in ('a', 'i', 'j', 't') for i in range(1, 4)
        ]
        self.file_mapping = negative + positive if n_class == 2 else negative + neutral + positive
        if len(self.file_mapping) != self.n_vids:
            raise ValueError(
                f"FACED feature mapping has {len(self.file_mapping)} videos, expected {self.n_vids}")

        # ================= 加载多模态特征 =================
        if sliced and image_feat_dir is not None and text_feat_dir is not None:
            self.has_multimodal = True
            print(f"[{mods}] Loading Multimodal Features...")

            self.video_feats = []
            self.text_feats = []

            # 遍历 28 个视频 (vid 0~27)
            for vid in range(self.n_vids):
                # 使用映射表找到对应的文件名
                fname = self.file_mapping[vid]

                v_path = os.path.join(image_feat_dir, fname)
                t_path = os.path.join(text_feat_dir, fname)

                if not os.path.exists(v_path):
                    raise FileNotFoundError(f"Missing IMAGE feature: {v_path}")
                if not os.path.exists(t_path):
                    raise FileNotFoundError(f"Missing TEXT feature: {t_path}")

                v_feat = np.load(v_path)
                t_feat = np.load(t_path)

                # 校验样本数是否匹配 (应该都是 13)
                if v_feat.shape[0] != self.samples_per_vid:
                    # 如果不匹配，尝试截取或填充，这里先报个警
                    print(
                        f"⚠️ Warning: Video {fname} (vid={vid}) feat len {v_feat.shape[0]} != EEG len {self.samples_per_vid}")

                self.video_feats.append(torch.from_numpy(v_feat).float())
                self.text_feats.append(torch.from_numpy(t_feat).float())

            print(f"[{mods}] Successfully loaded features for {len(self.video_feats)} videos.")

    def __len__(self):
        if self.mods == 'train' and self.train_subs is not None:
            return len(self.train_subs) * self.onesubLen
        elif self.mods == 'val' and self.val_subs is not None:
            return len(self.val_subs) * self.onesubLen
        else:
            return len(self.labels)  # 全量数据

    def __getitem__(self, idx):
        if self.mods == 'train':
            if self.train_subs is not None:
                idx = self.train_subs[idx // self.onesubLen] * self.onesubLen + idx % self.onesubLen
        elif self.mods == 'val':
            if self.val_subs is not None:
                idx = self.val_subs[idx // self.onesubLen] * self.onesubLen + idx % self.onesubLen

        # 加载 EEG
        one_seq = np.load(os.path.join(self.sliced_data_dir, 'data', f'data_sample_{idx}.npy'))
        one_seq = torch.FloatTensor(one_seq.reshape(1, one_seq.shape[-2], one_seq.shape[-1]))
        one_label = self.labels[idx]

        # 加载对应特征
        if self.has_multimodal:
            idx_in_sub = idx % self.onesubLen
            vid_id = idx_in_sub // self.samples_per_vid
            seg_id = idx_in_sub % self.samples_per_vid
            sub_id = idx // self.onesubLen           # 受试者 ID

            if vid_id >= self.n_vids:
                vid_id = self.n_vids - 1
            if seg_id >= self.samples_per_vid:
                seg_id = self.samples_per_vid - 1

            txt_feat = self.text_feats[vid_id][seg_id]
            img_feat = self.video_feats[vid_id][seg_id]

            return one_seq, one_label, txt_feat, img_feat, vid_id, sub_id
        else:
            return one_seq, one_label


SEED_FILE_MAPPING = [
    "positive/Lost_in_Thailand_1_features.npy",
    "neutral/Huangshan_1_features.npy",
    "negative/Tangshan_Earthquake_1_features.npy",
    "negative/1942_1_features.npy",
    "neutral/Huangshan_2_features.npy",
    "positive/Lost_in_Thailand_2_features.npy",
    "negative/1942_2_features.npy",
    "neutral/Suzhou_1_features.npy",
    "positive/Flirting_Scholar_features.npy",
    "positive/Just_Another_Pandoras_Box_1_features.npy",
    "neutral/Suzhou_2_features.npy",
    "negative/1942_3_features.npy",
    "neutral/Lijiang_1_features.npy",
    "positive/Just_Another_Pandoras_Box_2_features.npy",
    "negative/Tangshan_Earthquake_2_features.npy",
]


class SEED_Dataset_new(Dataset):
    """SEED EEG windows paired with the paper's fusion1 text/image features."""

    def __init__(self, load_dir, save_dir, timeLen, timeStep, train_subs=None, val_subs=None,
                 sliced=True, mods='train', n_session=3, fs=125, n_chans=62, n_subs=15,
                 n_vids=15, n_class=3, image_feat_dir=None, text_feat_dir=None):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.n_session = n_session
        self.fs = fs
        self.n_chans = n_chans
        self.n_subs = n_subs
        self.n_vids = n_vids
        self.n_class = n_class
        self.mods = mods
        if mods == 'train':
            self.active_subs = list(train_subs) if train_subs is not None else []
        else:
            self.active_subs = list(val_subs) if val_subs is not None else []
        self.sliced_data_dir = os.path.join(save_dir, f'sliced_len{timeLen}_step{timeStep}_SEED')

        if not sliced:
            marker_file = os.path.join(self.sliced_data_dir, 'saved.npy')
            if not os.path.exists(marker_file):
                data, labels, n_samples, n_samples_sessions = load_processed_SEED_NEW_data(
                    dir=load_dir, fs=fs, n_chans=n_chans, timeLen=timeLen, timeStep=timeStep,
                    n_session=n_session, n_subs=n_subs, n_vids=n_vids, n_class=n_class)
                save_sliced_data(self.sliced_data_dir, data, labels, n_samples, n_samples_sessions)
            return

        metadata_dir = os.path.join(self.sliced_data_dir, 'metadata')
        raw_labels = np.load(os.path.join(metadata_dir, 'onesub_labels.npy'))
        n_samples_all = np.load(os.path.join(metadata_dir, 'n_samples_onesub.npy'))
        n_segments = n_session * n_vids
        self.n_samples_original = n_samples_all[:n_segments].astype(int)
        self.cumulative_original = np.concatenate(([0], np.cumsum(self.n_samples_original)))
        self.onesub_len_original = int(self.n_samples_original.sum())
        if len(raw_labels) == self.onesub_len_original * n_subs:
            raw_labels = raw_labels[:self.onesub_len_original]
        elif len(raw_labels) != self.onesub_len_original:
            raise ValueError(
                f"SEED labels have length {len(raw_labels)}, expected "
                f"{self.onesub_len_original} or {self.onesub_len_original * n_subs}")

        has_image = image_feat_dir is not None
        has_text = text_feat_dir is not None
        if has_image != has_text:
            raise ValueError("SEED pretraining requires both text and image feature directories")
        self.has_multimodal = has_image and has_text

        if self.has_multimodal:
            self.image_features = []
            self.text_features = []
            for relative_path in SEED_FILE_MAPPING:
                image_path = os.path.join(image_feat_dir, relative_path)
                text_path = os.path.join(text_feat_dir, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Missing SEED image feature: {image_path}")
                if not os.path.exists(text_path):
                    raise FileNotFoundError(f"Missing SEED text feature: {text_path}")
                self.image_features.append(torch.from_numpy(np.load(image_path)).float())
                self.text_features.append(torch.from_numpy(np.load(text_path)).float())

            aligned_counts = []
            aligned_labels = []
            offset = 0
            for segment_idx, eeg_count in enumerate(self.n_samples_original):
                video_id = segment_idx % n_vids
                count = min(eeg_count, len(self.image_features[video_id]),
                            len(self.text_features[video_id]))
                aligned_counts.append(count)
                aligned_labels.append(raw_labels[offset:offset + count])
                offset += eeg_count
            self.n_samples_aligned = np.asarray(aligned_counts, dtype=int)
            self.onesub_labels = torch.from_numpy(np.concatenate(aligned_labels)).long()
        else:
            self.n_samples_aligned = self.n_samples_original
            self.onesub_labels = torch.from_numpy(raw_labels).long()

        self.cumulative_aligned = np.concatenate(([0], np.cumsum(self.n_samples_aligned)))
        self.onesubLen = int(self.n_samples_aligned.sum())

    def __len__(self):
        return len(self.active_subs) * self.onesubLen

    def __getitem__(self, idx):
        idx = int(idx)
        subject_position, aligned_offset = divmod(idx, self.onesubLen)
        subject_id = self.active_subs[subject_position]
        segment_idx = np.searchsorted(self.cumulative_aligned, aligned_offset, side='right') - 1
        offset_in_segment = aligned_offset - self.cumulative_aligned[segment_idx]
        original_idx = (subject_id * self.onesub_len_original
                        + self.cumulative_original[segment_idx] + offset_in_segment)
        eeg_path = os.path.join(self.sliced_data_dir, 'data', f'data_sample_{original_idx}.npy')
        eeg = np.load(eeg_path)
        eeg = torch.from_numpy(eeg).float().reshape(1, self.n_chans, -1)
        label = self.onesub_labels[aligned_offset]

        if not self.has_multimodal:
            return eeg, label

        video_id = segment_idx % self.n_vids
        text = self.text_features[video_id][offset_in_segment]
        image = self.image_features[video_id][offset_in_segment]
        content_id = int(video_id) * 1000 + int(offset_in_segment)
        return eeg, label, text, image, content_id, int(subject_id)
# ==========================================
# 其他类保持不变 (SEEDV, FACED_Dataset 旧版等)
# ==========================================

class FACED_Dataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data)  # n_samples * n_features
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        one_seq = self.data[idx].reshape(1, self.data.shape[-2], self.data.shape[-1])
        one_label = self.label[idx]
        return one_seq, one_label


class SEEDV_Dataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data)
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        one_seq = self.data[idx].reshape(1, self.data.shape[-2], self.data.shape[-1])
        one_label = self.label[idx]
        return one_seq, one_label


class SEEDV_Dataset_new(Dataset):
    def __init__(self, load_dir, save_dir, timeLen, timeStep, train_subs=None, val_subs=None, sliced=True, mods='train',
                 n_session=3, fs=125, n_chans=60, n_subs=16, n_vids=15, n_class=5):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.n_subs = n_subs
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.mods = mods
        self.sliced_data_dir = os.path.join(self.save_dir, f'sliced_len{self.timeLen}_step{self.timeStep}')
        self.load_processed_data = partial(load_processed_SEEDV_NEW_data, dir=self.load_dir,
                                           timeLen=self.timeLen, timeStep=self.timeStep)
        self.save_sliced_data = partial(save_sliced_data, sliced_data_dir=self.sliced_data_dir)
        if not sliced:
            if not os.path.exists(self.sliced_data_dir + '/saved.npy'):
                print('slicing processed dataset')
                data, onesub_labels, n_samples_onesub, n_samples_sessions = self.load_processed_data(
                    fs=fs, n_chans=n_chans, n_session=n_session, n_subs=n_subs, n_vids=n_vids, n_class=n_class)
                self.save_sliced_data(data=data, onesub_labels=onesub_labels, n_samples_onesub=n_samples_onesub,
                                      n_samples_sessions=n_samples_sessions)
            else:
                print('sliced data exist!')

        self.onesub_labels = torch.from_numpy(
            np.load(os.path.join(self.sliced_data_dir, 'metadata', 'onesub_labels.npy')))
        self.labels = self.onesub_labels.repeat(self.n_subs)
        self.onesubLen = len(self.onesub_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mods == 'train':
            if self.train_subs is not None:
                idx = self.train_subs[idx // self.onesubLen] * self.onesubLen + idx % self.onesubLen
        elif self.mods == 'val':
            if self.val_subs is not None:
                idx = self.val_subs[idx // self.onesubLen] * self.onesubLen + idx % self.onesubLen
        one_seq = np.load(os.path.join(self.sliced_data_dir, 'data', f'data_sample_{idx}.npy'))
        one_seq = torch.FloatTensor(
            one_seq.reshape(1, one_seq.shape[-2], one_seq.shape[-1]))  # 32*(250)->1*32*250  2d conv  c*h*w
        one_label = self.labels[idx]
        return one_seq, one_label


class EEG_Dataset(Dataset):
    def __init__(self, cfg, train_subs=None, val_subs=None, sliced=True, mods=None):
        self.load_dir = os.path.join(cfg.data_dir, 'processed_data')
        self.save_dir = os.path.join(cfg.data_dir, 'sliced_data')

        self.train_subs = train_subs
        self.val_subs = val_subs
        self.mods = mods

        self.sliced_data_dir = os.path.join(self.save_dir, f'sliced_len{cfg.timeLen}_step{cfg.timeStep}')

        if not sliced:
            if not os.path.exists(self.sliced_data_dir + '/saved.npy'):
                print('slicing processed dataset')
                data, onesub_labels, n_samples_onesub, n_samples_sessions = load_EEG_data(self.load_dir, cfg)
                save_sliced_data(sliced_data_dir=self.sliced_data_dir, data=data, onesub_labels=onesub_labels,
                                 n_samples_onesub=n_samples_onesub, n_samples_sessions=n_samples_sessions)
            else:
                print('sliced data exist!')

        self.onesub_labels = torch.from_numpy(
            np.load(os.path.join(self.sliced_data_dir, 'metadata', 'onesub_labels.npy')))
        self.labels = self.onesub_labels.repeat(cfg.n_subs)
        self.onesubLen = len(self.onesub_labels)

    def __len__(self):
        if self.mods == 'train':
            if self.train_subs is not None:
                return len(self.train_subs) * self.onesubLen
        elif self.mods == 'val':
            if self.val_subs is not None:
                return len(self.val_subs) * self.onesubLen
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        if self.mods == 'train':
            if self.train_subs is not None:
                idx = self.train_subs[idx // self.onesubLen] * self.onesubLen + idx % self.onesubLen
        elif self.mods == 'val':
            if self.val_subs is not None:
                idx = self.val_subs[idx // self.onesubLen] * self.onesubLen + idx % self.onesubLen
        one_seq = np.load(os.path.join(self.sliced_data_dir, 'data', f'data_sample_{idx}.npy'))
        one_seq = torch.FloatTensor(one_seq.reshape(1, one_seq.shape[-2], one_seq.shape[-1]))
        one_label = self.labels[idx]
        return one_seq, one_label


class PDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data)
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        one_seq = self.data[idx]
        one_label = self.label[idx]
        return one_seq, one_label


# ... (保持原本的 Sampler 类不变) ...
class TrainSampler_FACED():
    def __init__(self, n_subs, batch_size, n_samples, n_session=1, n_times=1):
        self.n_per = int(np.sum(n_samples))
        self.n_subs = n_subs
        self.batch_size = batch_size
        self.n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
        self.n_samples_per_trial = int(batch_size / len(n_samples))
        self.sub_pairs = []
        for i in range(self.n_subs):
            for j in range(i + n_session, self.n_subs, n_session):
                self.sub_pairs.append([i, j])
        random.shuffle(self.sub_pairs)
        self.n_times = n_times

    def __len__(self):
        return self.n_times * len(self.sub_pairs)

    def __iter__(self):
        for s in range(len(self.sub_pairs)):
            for t in range(self.n_times):
                [sub1, sub2] = self.sub_pairs[s]
                ind_abs = np.zeros(0)
                if self.batch_size < len(self.n_samples_cum) - 1:
                    sel_vids = np.random.choice(np.arange(len(self.n_samples_cum) - 1), self.batch_size)
                    for i in sel_vids:
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i + 1]), 1,
                                                   replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                else:
                    for i in range(len(self.n_samples_cum) - 2):
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i + 1]),
                                                   self.n_samples_per_trial, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                    i = len(self.n_samples_cum) - 2
                    ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i + 1]),
                                               int(self.batch_size - len(ind_abs)), replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))
                assert len(ind_abs) == self.batch_size
                ind_this1 = ind_abs + self.n_per * sub1
                ind_this2 = ind_abs + self.n_per * sub2
                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                yield batch


class TrainSampler_SEEDV():
    def __init__(self, n_subs, batch_size, n_samples_session, n_session=1, n_times=1, if_val_loo=False):
        self.n_per_session = np.sum(n_samples_session, 1).astype(int)
        self.n_per_session_cum = np.concatenate((np.array([0]), np.cumsum(self.n_per_session)))
        self.n_subs = n_subs
        self.batch_size = batch_size
        self.n_samples_cum_session = np.concatenate((np.zeros((n_session, 1)), np.cumsum(n_samples_session, 1)), 1)
        self.n_samples_per_trial = int(batch_size / n_samples_session.shape[1])
        self.subsession_pairs = []
        self.n_session = n_session
        if if_val_loo:
            self.n_pairsubs = 1
        else:
            self.n_pairsubs = self.n_subs
        for i in range(self.n_pairsubs * self.n_session):
            for j in range(i + n_session, self.n_subs * self.n_session, n_session):
                self.subsession_pairs.append([i, j])
        random.shuffle(self.subsession_pairs)
        self.n_times = n_times

    def __len__(self):  # n_batch
        return self.n_times * len(self.subsession_pairs)

    def __iter__(self):
        for s in range(len(self.subsession_pairs)):
            for t in range(self.n_times):
                [subsession1, subsession2] = self.subsession_pairs[s]
                cur_session = int(subsession1 % self.n_session)
                cur_sub1 = int(subsession1 // self.n_session)
                cur_sub2 = int(subsession2 // self.n_session)

                ind_abs = np.zeros(0)
                if self.batch_size < len(self.n_samples_cum_session[cur_session]) - 1:
                    sel_vids = np.random.choice(np.arange(len(self.n_samples_cum_session[cur_session]) - 1),
                                                self.batch_size)
                    for i in sel_vids:
                        ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i],
                                                             self.n_samples_cum_session[cur_session][i + 1]), 1,
                                                   replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                else:
                    for i in range(len(self.n_samples_cum_session[cur_session]) - 2):
                        ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i],
                                                             self.n_samples_cum_session[cur_session][i + 1]),
                                                   self.n_samples_per_trial, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                    i = len(self.n_samples_cum_session[cur_session]) - 2
                    ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i],
                                                         self.n_samples_cum_session[cur_session][i + 1]),
                                               int(self.batch_size - len(ind_abs)), replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))
                assert len(ind_abs) == self.batch_size
                ind_this1 = ind_abs + np.sum(self.n_per_session) * cur_sub1 + self.n_per_session_cum[cur_session]
                ind_this2 = ind_abs + np.sum(self.n_per_session) * cur_sub2 + self.n_per_session_cum[cur_session]
                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                yield batch


class PretrainSampler():
    def __init__(self, n_subs, batch_size, n_samples_session, n_times=1, if_val_loo=False,
                 cross_session=False):
        self.n_per_session = np.sum(n_samples_session, 1).astype(int)
        self.n_per_session_cum = np.concatenate((np.array([0]), np.cumsum(self.n_per_session)))
        self.n_subs = n_subs
        self.batch_size = batch_size
        self.n_samples_per_trial = int(batch_size / n_samples_session.shape[1])
        self.subsession_pairs = []
        self.n_session = n_samples_session.shape[0]
        self.n_samples_cum_session = np.concatenate((np.zeros((self.n_session, 1)), np.cumsum(n_samples_session, 1)), 1)
        if if_val_loo:
            self.n_pairsubs = 1
        else:
            self.n_pairsubs = self.n_subs
        if cross_session:
            total_sub_sessions = self.n_subs * self.n_session
            for i in range(total_sub_sessions):
                for j in range(i + 1, total_sub_sessions):
                    self.subsession_pairs.append([i, j])
        else:
            for i in range(self.n_pairsubs * self.n_session):
                for j in range(i + self.n_session, self.n_subs * self.n_session, self.n_session):
                    self.subsession_pairs.append([i, j])
        random.shuffle(self.subsession_pairs)
        self.n_times = n_times

    def __len__(self):
        return self.n_times * len(self.subsession_pairs)

    def __iter__(self):
        for s in range(len(self.subsession_pairs)):
            for t in range(self.n_times):
                [subsession1, subsession2] = self.subsession_pairs[s]
                cur_session = int(subsession1 % self.n_session)
                cur_sub1 = int(subsession1 // self.n_session)
                cur_sub2 = int(subsession2 // self.n_session)

                ind_abs = np.zeros(0)
                if self.batch_size < len(self.n_samples_cum_session[cur_session]) - 1:
                    sel_vids = np.random.choice(np.arange(len(self.n_samples_cum_session[cur_session]) - 1),
                                                self.batch_size)
                    for i in sel_vids:
                        ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i],
                                                             self.n_samples_cum_session[cur_session][i + 1]), 1,
                                                   replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                else:
                    for i in range(len(self.n_samples_cum_session[cur_session]) - 2):
                        ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i],
                                                             self.n_samples_cum_session[cur_session][i + 1]),
                                                   self.n_samples_per_trial, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                    i = len(self.n_samples_cum_session[cur_session]) - 2
                    ind_one = np.random.choice(np.arange(self.n_samples_cum_session[cur_session][i],
                                                         self.n_samples_cum_session[cur_session][i + 1]),
                                               int(self.batch_size - len(ind_abs)), replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))
                assert len(ind_abs) == self.batch_size
                ind_this1 = ind_abs + np.sum(self.n_per_session) * cur_sub1 + self.n_per_session_cum[cur_session]
                ind_this2 = ind_abs + np.sum(self.n_per_session) * cur_sub2 + self.n_per_session_cum[cur_session]
                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                yield batch


if __name__ == '__main__':
    pass
