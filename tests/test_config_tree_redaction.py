"""
(c) Inria

The config tree is printed on **every** run — ``configs/extras/default.yaml`` sets
``print_config: true`` — with ``resolve=True`` and ``save_to_file=True``. So whatever it
renders is both echoed to the terminal and written to ``config_tree.log`` in the output
dir, where on CI or a shared HPC filesystem it outlives the run.

These tests assert the one property that makes that safe: **a credential never reaches
the rendered tree**, in either destination.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pytest
from omegaconf import OmegaConf

from planktonzilla.utils.rich_utils import REDACTED, print_config_tree, redact_secrets

TOKEN = "hf_SECRET_TOKEN_ABC123"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """The shape the real configs have: the token arrives by env interpolation."""
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    return OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "dataset_import": {"hf_token": "${oc.env:HF_TOKEN, null}", "repo_id": "project-oceania/planktonzilla-17M"},
        }
    )


def test_the_token_reaches_neither_the_terminal_nor_the_log(cfg, tmp_path, capsys):
    """The finding, end to end: resolve=True expanded it into both destinations."""
    print_config_tree(cfg, resolve=True, save_to_file=True)

    printed = capsys.readouterr().out
    saved = (tmp_path / "config_tree.log").read_text(encoding="utf-8")

    assert TOKEN not in printed, "the live token was echoed to the terminal"
    assert TOKEN not in saved, "the live token was persisted to config_tree.log"
    assert REDACTED in printed and REDACTED in saved
    # The control: the tree is still a tree, and the non-secret value still renders.
    assert "project-oceania" in printed


def test_the_config_the_job_runs_on_is_not_touched(cfg):
    """Redaction is on the rendered copy only — masking the live config would break auth."""
    print_config_tree(cfg, resolve=True, save_to_file=False)

    assert cfg.dataset_import.hf_token == TOKEN


def test_no_token_set_reads_as_null_rather_than_as_a_mask(tmp_path, monkeypatch):
    """ "There is no token" is worth seeing; <redacted> over nothing states the opposite."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = OmegaConf.create({"paths": {"output_dir": str(tmp_path)}, "hf_token": "${oc.env:HF_TOKEN, null}"})

    assert redact_secrets(cfg).hf_token is None


@pytest.mark.parametrize("key", ["hf_token", "token", "hub_token", "api_key", "password", "client_secret"])
def test_credential_keys_are_masked(key):
    assert redact_secrets(OmegaConf.create({key: "sensitive"}))[key] == REDACTED


@pytest.mark.parametrize("key", ["tokenizer", "include_tokens_per_second", "tokens_per_batch", "secretariat"])
def test_keys_that_merely_contain_a_secret_word_are_not_masked(key):
    """`tokenizer` and `include_tokens_per_second` are real keys in configs/training_arguments.

    A substring match would blank them, which turns a safety feature into a config tree
    that no longer says what the run is doing.
    """
    assert redact_secrets(OmegaConf.create({key: "visible"}))[key] == "visible"


def test_a_credential_nested_in_a_list_is_masked():
    """Nothing in the tree is exempt by position."""
    cfg = OmegaConf.create({"remotes": [{"name": "hub", "api_key": "sensitive"}]})

    assert redact_secrets(cfg).remotes[0].api_key == REDACTED
    assert redact_secrets(cfg).remotes[0].name == "hub"
