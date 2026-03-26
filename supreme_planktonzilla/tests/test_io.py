"""Tests for utils/io.py — load_yaml, save_json, config_stem."""

import json
import os

import pytest

from utils.io import config_stem, load_yaml, save_json


# ── config_stem ───────────────────────────────────────────────────────────────

def test_config_stem_simple():
    assert config_stem("config/supreme_default.yaml") == "supreme_default"


def test_config_stem_no_directory():
    assert config_stem("myconfig.yaml") == "myconfig"


def test_config_stem_deep_path():
    assert config_stem("/a/b/c/experiment_v2.yaml") == "experiment_v2"


# ── load_yaml ─────────────────────────────────────────────────────────────────

def test_load_yaml_returns_dict(tmp_path):
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("lr: 0.001\nepochs: 50\ndevice: cuda\n")
    data = load_yaml(str(yaml_file))
    assert data == {"lr": 0.001, "epochs": 50, "device": "cuda"}


def test_load_yaml_list_value(tmp_path):
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("ood_roots:\n  - /data/ood1\n  - /data/ood2\n")
    data = load_yaml(str(yaml_file))
    assert data["ood_roots"] == ["/data/ood1", "/data/ood2"]


def test_load_yaml_real_config():
    """The committed supreme_default.yaml must load without errors."""
    data = load_yaml("config/supreme_default.yaml")
    assert "clip_model" in data
    assert "epochs" in data
    assert "lr" in data


# ── save_json ─────────────────────────────────────────────────────────────────

def test_save_json_creates_file(tmp_path):
    path = str(tmp_path / "results" / "out.json")
    save_json({"auroc": 0.95, "fpr95": 12.3}, path)
    assert os.path.exists(path)


def test_save_json_content_is_valid(tmp_path):
    path = str(tmp_path / "out.json")
    data = {"trial0": {"top1_acc": 88.5, "ood": {"dataset1": {"fpr95": 10.0}}}}
    save_json(data, path)
    loaded = json.loads(open(path).read())
    assert loaded == data


def test_save_json_creates_nested_dirs(tmp_path):
    path = str(tmp_path / "a" / "b" / "c" / "results.json")
    save_json({"ok": True}, path)
    assert os.path.exists(path)
