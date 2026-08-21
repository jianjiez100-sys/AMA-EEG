import tempfile
import unittest
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from FACED.move_window_timelen5_timestep2_EEG import process_eeg


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_dataset_configs_compose_with_relative_paths(self):
        with initialize_config_dir(config_dir=str(ROOT / "cfgs"), version_base="1.3"):
            for dataset, expected_classes in (("FACED", 9), ("FACED_def_c2", 2), ("SEED", 3)):
                cfg = compose(config_name="config", overrides=[f"data={dataset}"])
                OmegaConf.resolve(cfg)
                self.assertEqual(cfg.data.n_class, expected_classes)
                self.assertFalse(Path(cfg.data.data_dir).is_absolute())
                self.assertFalse(Path(cfg.data.text_feat_dir).is_absolute())
                self.assertFalse(Path(cfg.data.image_feat_dir).is_absolute())

    def test_faced_windowing_with_small_synthetic_input(self):
        n_subs, n_vids, seconds, feat_dim = 2, 3, 6, 4
        raw = np.arange(n_subs * n_vids * seconds * feat_dim, dtype=np.float32)
        raw = raw.reshape(-1, feat_dim)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.npy"
            np.save(input_path, raw)
            output_path = process_eeg(
                input_path,
                tmp_path / "output",
                n_subs=n_subs,
                n_vids=n_vids,
                seconds=seconds,
                feat_dim=feat_dim,
                window_size=3,
                stride=2,
            )
            result = np.load(output_path)
            self.assertEqual(result.shape, (12, 3, 4))

    def test_required_repository_documents_exist(self):
        for name in (
            "README.md",
            "LICENSE",
            "CITATION.cff",
            "DATA.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "requirements.txt",
        ):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_citation_metadata_links_the_accepted_paper(self):
        citation = OmegaConf.load(ROOT / "CITATION.cff")
        self.assertEqual(citation.title, "AMA-EEG")
        self.assertEqual(len(citation.authors), 6)
        self.assertEqual(
            citation["preferred-citation"].journal,
            "IEEE Transactions on Affective Computing",
        )


if __name__ == "__main__":
    unittest.main()
