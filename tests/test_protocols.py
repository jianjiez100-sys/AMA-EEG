"""Regression checks for differences that affect the scientific pipeline."""
from pathlib import Path
import pickle
import tempfile
import unittest

from hydra import compose, initialize_config_dir
import numpy as np
from scipy.io import savemat
from scipy.signal import resample
import torch
from omegaconf import OmegaConf

from data.dataset import PretrainSampler, SEED_Dataset_new, SEED_FILE_MAPPING
from data.feature_processing import normalize_extraction_input, smooth_trial_features
from data.io_utils import load_processed_SEED_NEW_data, save_sliced_data
from model import get_extractor_class
from model.loss.seed_con_loss import MultiModalInfoNCELoss
from extract_features import extraction_labels, load_extraction_data
from utils.reorder_vids import video_order_load, reorder_vids_sepVideo, reorder_vids_back

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def _check_public_fusion(self, stage):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, eeg, **kwargs):
                return eeg * self.scale

            def set_saveFea(self, enabled):
                pass

        class FixedProbe(torch.nn.Module):
            def __init__(self, probabilities):
                super().__init__()
                self.register_buffer('logits', probabilities.log())

            def forward(self, features):
                return self.logits

        class CaptureAlignment(torch.nn.Module):
            def forward(self, eeg, target, **kwargs):
                self.target = target.detach().clone()
                self.identifiers = kwargs
                return (eeg - target).square().mean()

        labels = torch.tensor([0, 0, 0, 0, 1, 1])
        videos = torch.tensor([0, 0, 1, 1, 2, 2])
        subjects = torch.tensor([0, 1, 0, 1, 0, 1])
        # Same-class videos get separate FACED weights and a shared SEED weight.
        text_true = torch.tensor([0.8, 0.5, 0.2, 0.3, 0.4, 0.7])
        image_true = torch.tensor([0.2, 0.5, 0.8, 0.6, 0.6, 0.2])
        probabilities = []
        for true_probability in (text_true, image_true):
            values = torch.full((6, 3), 0.1)
            values[torch.arange(6), labels] = true_probability
            values[torch.arange(6), 1 - labels] = 0.9 - true_probability
            probabilities.append(values)
        text = torch.zeros(6, 1024)
        image = torch.zeros_like(text)
        text[:, 0] = 1
        image[:, 1] = 1
        eeg = torch.eye(6, 1024)

        # At temperature 1, CE-based alpha simplifies to p_text / (p_text+p_image).
        # This oracle uses true-class probabilities, independently of the CE code.
        instant = text_true / (text_true + image_true)
        for dataset in ('FACED', 'SEED'):
            with self.subTest(stage=stage, dataset=dataset):
                if stage == 'validation':
                    expected_alpha = instant
                elif dataset == 'FACED':
                    expected_alpha = torch.tensor([
                        (instant[0] + instant[1]) / 2, (instant[0] + instant[1]) / 2,
                        (instant[2] + instant[3]) / 2, (instant[2] + instant[3]) / 2,
                        (instant[4] + instant[5]) / 2, (instant[4] + instant[5]) / 2,
                    ])
                else:
                    expected_alpha = torch.cat((instant[:4].mean().repeat(4),
                                                instant[4:].mean().repeat(2)))
                expected_target = torch.nn.functional.normalize(
                    expected_alpha[:, None] * text + (1 - expected_alpha[:, None]) * image,
                    dim=1)
                with initialize_config_dir(config_dir=str(ROOT / 'cfgs'), version_base='1.3'):
                    cfg = compose(config_name='config', overrides=[f'data={dataset}'])
                cfg.train.use_two_stage_projector = False
                cfg.train.n_class = 3
                module = get_extractor_class(dataset)(Backbone(), cfg.train)
                module.conf_temp = 1.0
                module.text_probe = FixedProbe(probabilities[0])
                module.image_probe = FixedProbe(probabilities[1])
                module.text_projector = torch.nn.Identity()
                module.image_projector = torch.nn.Identity()
                alignment = CaptureAlignment()
                module.distill_criterion = alignment
                module.log = lambda *args, **kwargs: None
                module.log_dict = lambda *args, **kwargs: None
                module.optimizers = lambda: type('Optimizer', (), {'param_groups': [{'lr': 0.001}]})()
                content = videos * 1000 + torch.tensor([11, 19, 11, 19, 7, 13])
                extras = (subjects * 100000 + content, content) if dataset == 'SEED' else (videos, subjects)
                getattr(module, f'{stage}_step')((eeg, labels, text, image, *extras), 0)
                torch.testing.assert_close(alignment.target, expected_target, rtol=1e-6, atol=1e-7)
                if dataset == 'SEED':
                    torch.testing.assert_close(alignment.identifiers['content_ids'], content)

    def test_training_fusion_uses_dataset_specific_groups(self):
        self._check_public_fusion('training')

    def test_validation_fusion_uses_individual_ce_alpha(self):
        self._check_public_fusion('validation')

    def test_dataset_defaults_preserve_different_experiments(self):
        with initialize_config_dir(config_dir=str(ROOT / 'cfgs'), version_base='1.3'):
            faced = compose(config_name='config', overrides=['data=FACED'])
            seed = compose(config_name='config', overrides=['data=SEED'])
            binary = compose(config_name='config', overrides=['data=FACED_def_c2'])
        self.assertEqual((faced.train.max_epochs, faced.train.min_epochs, faced.train.patience), (15, 3, 3))
        self.assertEqual((seed.train.max_epochs, seed.train.min_epochs, seed.train.patience), (25, 10, 5))
        self.assertEqual((faced.train.probe_loss_weight, seed.train.probe_loss_weight), (2, 1))
        self.assertEqual((faced.mlp.lr, seed.mlp.lr), (0.0002, 0.0005))
        self.assertTrue(faced.train.use_two_stage_projector)
        self.assertFalse(seed.train.use_two_stage_projector)
        self.assertEqual(binary.data.n_vids, 28)
        self.assertEqual(len(PretrainSampler(14, 15, np.ones((3, 15), dtype=int), seed.train.sampler_times)), 2730)

    def test_seed_raw_trials_and_multimodal_indices(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '1').mkdir()
            raw_trials = []
            for subject in range(2):
                # Unequal lengths exercise truncation to the shortest subject.
                trial = rng.normal(size=(2, (7 + subject * 2) * 200))
                trial[1] = trial[1] * 12 + 30
                raw_trials.append(trial)
                savemat(root / '1' / f'{subject + 1}_fixture.mat',
                        {f'fixture_eeg{video + 1}': trial + video for video in range(15)})
            data, labels, counts, sessions = load_processed_SEED_NEW_data(
                str(root), 125, 2, 5, 2, n_session=1, n_subs=2)
            self.assertEqual(data.shape, (60, 2, 625))
            np.testing.assert_array_equal(sessions, np.full((1, 15), 2))
            for subject, trial in enumerate(raw_trials):
                sampled = resample(trial, int(trial.shape[1] * 125 / 200), axis=1)
                expected = (sampled - sampled.mean(1, keepdims=True)) / (sampled.std(1, keepdims=True) + 1e-8)
                np.testing.assert_allclose(data[subject * 30], expected[:, :625], rtol=0, atol=0)
                np.testing.assert_allclose(data[subject * 30 + 1], expected[:, 250:875], rtol=0, atol=0)
            cache = root / 'sliced_len5_step2_SEED'
            save_sliced_data(str(cache), data, labels, counts, sessions)
            for modality in ('text', 'image'):
                for video, relative in enumerate(SEED_FILE_MAPPING):
                    path = root / modality / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Shorter stimulus features exercise EEG/teacher alignment.
                    np.save(path, np.full((1, 5, 1024), video, dtype=np.float32))
            dataset = SEED_Dataset_new(
                str(root), str(root), 5, 2, train_subs=[0, 1],
                n_session=1, n_chans=2, n_subs=2,
                text_feat_dir=str(root / 'text'), image_feat_dir=str(root / 'image'))
            self.assertEqual(len(dataset), 30)
            first, second = dataset[3], dataset[18]
            torch.testing.assert_close(first[0][0], torch.from_numpy(data[6]).float())
            torch.testing.assert_close(second[0][0], torch.from_numpy(data[36]).float())
            self.assertEqual(first[-1], second[-1])
            self.assertNotEqual(first[-2], second[-2])
            self.assertEqual(first[-1], 3000)
            # The earlier release's one-subject metadata must not silently reuse
            # EEG normalized by the wrong merged-MAT preprocessing rule.
            np.save(cache / 'metadata' / 'onesub_labels.npy', labels[:30])
            with self.assertRaisesRegex(ValueError, 'fresh cache directory'):
                SEED_Dataset_new(str(root), str(root), 5, 2, train_subs=[0, 1],
                                 n_session=1, n_chans=2, n_subs=2)

    def test_fold_normalization_uses_only_training_subjects(self):
        data = np.random.default_rng(7).normal(size=(3, 6, 4, 9)).astype(np.float32)
        changed = data.copy()
        changed[0] = changed[0] * 100 + 200
        for dataset in ('FACED', 'SEED'):
            expected = normalize_extraction_input(data, [1, 2], dataset)
            actual = normalize_extraction_input(changed, [1, 2], dataset)
            np.testing.assert_array_equal(expected[1:], actual[1:])
        faced = normalize_extraction_input(data, [1, 2], 'FACED')
        np.testing.assert_allclose(faced[1:].mean(axis=(0, 1, 3)), 0, atol=1e-6)

    def test_seed_minimum_is_separate_for_every_session_and_video(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_counts = np.stack([2 + np.arange(15) % 3, 3 + np.arange(15) % 3])
            for session in range(2):
                (root / str(session + 1)).mkdir()
                for subject in range(2):
                    trials = {}
                    for video in range(15):
                        # Different subjects, videos, and sessions have different lengths.
                        seconds = 7 + 2 * (video % 3) + 2 * session + 2 * subject
                        trials[f'fixture_eeg{video + 1}'] = rng.normal(size=(2, seconds * 200))
                    savemat(root / str(session + 1) / f'{subject + 1}_fixture.mat', trials)
            data, labels, counts, sessions = load_processed_SEED_NEW_data(
                str(root), 125, 2, 5, 2, n_session=2, n_subs=2)
            np.testing.assert_array_equal(sessions, expected_counts)
            self.assertEqual(data.shape, (2 * expected_counts.sum(), 2, 625))
            np.testing.assert_array_equal(counts, np.tile(expected_counts.flatten(), 2))
            label_order = np.array([2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0])
            one_subject = np.concatenate([np.repeat(label_order, row) for row in expected_counts])
            np.testing.assert_array_equal(labels, np.tile(one_subject, 2))

    def test_extraction_respects_faced_pkl_and_seed_shard_formats(self):
        rng = np.random.default_rng(13)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            faced_dir = root / 'FACED' / 'all_pkl_files'
            faced_dir.mkdir(parents=True)
            for subject in range(2):
                with (faced_dir / f'sub{subject:03}.pkl').open('wb') as output:
                    pickle.dump(rng.normal(size=(28, 2, 7500)).astype(np.float32), output)
            faced_cfg = OmegaConf.create({'data': dict(
                dataset_name='FACED', data_dir=str(faced_dir.parent), fs=125,
                n_channs=2, n_subs=2, n_session=1, n_vids=28, n_class=2,
                timeLen2=5, timeStep2=2)})
            faced, faced_labels, faced_counts = load_extraction_data(faced_cfg)
            self.assertEqual(faced.shape, (2, 364, 2, 625))
            np.testing.assert_array_equal(faced_counts, np.full((1, 28), 13))
            self.assertEqual(set(faced_labels), set(range(9)))
            self.assertEqual(extraction_labels(faced_labels, 2, 364, 'FACED').shape, (728,))

            seed_dir = root / 'SEED'
            cache = seed_dir / 'sliced_len5_step2_SEED'
            (cache / 'split_by_sub').mkdir(parents=True)
            seed_counts = np.stack([2 + np.arange(15) % 3, 3 + np.arange(15) % 3])
            n_windows = int(seed_counts.sum())
            seed_arrays = [rng.normal(size=(n_windows, 2, 625)).astype(np.float32) for _ in range(2)]
            for subject, data in enumerate(seed_arrays):
                np.save(cache / 'split_by_sub' / f'sub{subject:03}_data.npy', data)
            full_labels = np.arange(2 * n_windows) % 3
            np.save(cache / 'onesub_labels.npy', full_labels)
            np.save(cache / 'n_samples_sessions.npy', seed_counts)
            seed_cfg = OmegaConf.create({'data': dict(
                dataset_name='SEED', data_dir=str(seed_dir), fs=125,
                n_channs=2, n_subs=2, n_session=2, n_vids=15, n_class=3,
                timeLen2=5, timeStep2=2)})
            seed, seed_labels, loaded_counts = load_extraction_data(seed_cfg)
            np.testing.assert_array_equal(seed, np.stack(seed_arrays))
            np.testing.assert_array_equal(loaded_counts, seed_counts)
            np.testing.assert_array_equal(seed_labels, full_labels)
            np.testing.assert_array_equal(
                extraction_labels(seed_labels, 2, n_windows, 'SEED'), full_labels)
            with self.assertRaises(ValueError):
                extraction_labels(seed_labels[:-1], 2, n_windows, 'SEED')

    def test_lds_does_not_cross_trial_boundaries(self):
        features = np.random.default_rng(7).normal(size=(2, 6, 5)).astype(np.float32)
        changed = features.copy()
        changed[:, 2:] += 100
        first = smooth_trial_features(features, [2, 4])
        second = smooth_trial_features(changed, [2, 4])
        np.testing.assert_array_equal(first[:, :2], second[:, :2])
        with self.assertRaises(ValueError):
            smooth_trial_features(features, [2, 3])

    def test_faced_playback_order_roundtrip(self):
        orders = video_order_load(28)
        expected_ids = np.broadcast_to(np.arange(1, 29), orders.shape)
        np.testing.assert_array_equal(np.sort(orders, axis=1), expected_ids)
        features = np.arange(2 * 56 * 3).reshape(2, 56, 3).astype(np.float32)
        reordered, play_order = reorder_vids_sepVideo(features, orders[:2], np.arange(28), 28)
        restored = reorder_vids_back(reordered, 28, play_order)
        np.testing.assert_array_equal(restored, features)

    def test_same_content_is_excluded_as_a_negative(self):
        features = torch.eye(3)
        loss = MultiModalInfoNCELoss(temperature=0.1)
        unmasked = loss(features, features)
        masked = loss(features, features, content_ids=torch.tensor([0, 0, 1]))
        self.assertLess(masked.item(), unmasked.item())
        self.assertTrue(torch.isfinite(masked))


if __name__ == '__main__':
    unittest.main()
