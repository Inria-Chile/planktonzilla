"""Tests for supreme/config.py — Config dataclass and from_yaml()."""

import pytest

from supreme.config import Config
from utils.datasets import DatasetConfig


class TestConfigDefaults:
    def test_default_clip_model(self):
        cfg = Config()
        assert cfg.clip_model == "hf-hub:imageomics/bioclip-2"

    def test_default_embed_dim(self):
        assert Config().embed_dim == 768

    def test_default_shots(self):
        assert Config().shots == 16

    def test_default_epochs(self):
        assert Config().epochs == 50

    def test_default_score(self):
        assert Config().score == "gmp"

    def test_default_id_train_is_dataset_config(self):
        cfg = Config()
        assert isinstance(cfg.id_train, DatasetConfig)
        assert cfg.id_train.source == ""
        assert cfg.id_train.split == "train"
        assert cfg.id_train.samples_per_class is None

    def test_default_id_test_is_dataset_config(self):
        cfg = Config()
        assert isinstance(cfg.id_test, DatasetConfig)

    def test_default_ood_test_is_empty_list(self):
        cfg = Config()
        assert cfg.ood_test == []
        # Must be a new list per instance (mutable default)
        cfg2 = Config()
        cfg.ood_test.append(DatasetConfig())
        assert cfg2.ood_test == []


class TestConfigFromYaml:
    def test_loads_real_yaml(self):
        cfg = Config.from_yaml("config/supreme_default.yaml")
        assert isinstance(cfg, Config)

    def test_overrides_scalar_value(self, tmp_path):
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text("epochs: 100\nlr: 0.005\n")
        cfg = Config.from_yaml(str(yaml_file))
        assert cfg.epochs == 100
        assert cfg.lr == pytest.approx(0.005)

    def test_unspecified_keys_keep_defaults(self, tmp_path):
        yaml_file = tmp_path / "partial.yaml"
        yaml_file.write_text("epochs: 10\n")
        cfg = Config.from_yaml(str(yaml_file))
        assert cfg.shots == 16
        assert cfg.embed_dim == 768

    def test_unknown_yaml_keys_are_ignored(self, tmp_path):
        yaml_file = tmp_path / "extra.yaml"
        yaml_file.write_text("epochs: 5\nsome_unknown_key: 999\n")
        cfg = Config.from_yaml(str(yaml_file))
        assert cfg.epochs == 5

    def test_id_train_parsed_as_dataset_config(self, tmp_path):
        yaml_file = tmp_path / "ds.yaml"
        yaml_file.write_text(
            "id_train:\n"
            "  source: project-oceania/planktonzilla\n"
            "  split: train\n"
            "  samples_per_class: 50\n"
        )
        cfg = Config.from_yaml(str(yaml_file))
        assert isinstance(cfg.id_train, DatasetConfig)
        assert cfg.id_train.source == "project-oceania/planktonzilla"
        assert cfg.id_train.split == "train"
        assert cfg.id_train.samples_per_class == 50

    def test_id_test_parsed_as_dataset_config(self, tmp_path):
        yaml_file = tmp_path / "ds.yaml"
        yaml_file.write_text(
            "id_test:\n"
            "  source: project-oceania/planktonzilla\n"
            "  split: test\n"
            "  samples_per_class:\n"
        )
        cfg = Config.from_yaml(str(yaml_file))
        assert isinstance(cfg.id_test, DatasetConfig)
        assert cfg.id_test.split == "test"
        assert cfg.id_test.samples_per_class is None

    def test_ood_test_single_block_becomes_list(self, tmp_path):
        yaml_file = tmp_path / "ood.yaml"
        yaml_file.write_text(
            "ood_test:\n"
            "  source: /data/ood_dataset\n"
            "  split: train\n"
            "  samples_per_class:\n"
        )
        cfg = Config.from_yaml(str(yaml_file))
        assert isinstance(cfg.ood_test, list)
        assert len(cfg.ood_test) == 1
        assert isinstance(cfg.ood_test[0], DatasetConfig)
        assert cfg.ood_test[0].source == "/data/ood_dataset"

    def test_ood_test_list_of_blocks(self, tmp_path):
        yaml_file = tmp_path / "ood_multi.yaml"
        yaml_file.write_text(
            "ood_test:\n"
            "  - source: /data/ood1\n"
            "    split: train\n"
            "    samples_per_class:\n"
            "  - source: /data/ood2\n"
            "    split: test\n"
            "    samples_per_class: 100\n"
        )
        cfg = Config.from_yaml(str(yaml_file))
        assert len(cfg.ood_test) == 2
        assert cfg.ood_test[0].source == "/data/ood1"
        assert cfg.ood_test[1].source == "/data/ood2"
        assert cfg.ood_test[1].samples_per_class == 100

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("nonexistent/path.yaml")
