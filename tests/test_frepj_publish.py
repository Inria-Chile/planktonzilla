"""
(c) Inria

Network-free tests for the FREPJ-only publish helper (Plan 19-02, VAL-01).

They pin, by construction (NO real push, NO network — every hub call is a fake or is
rejected before it reaches the wire):

  (a) ``preflight`` rejects the frozen ``planktonzilla-17M`` (full id + bare basename) and
      ANY non-target repo, and allows ONLY ``project-oceania/planktonzilla-frepj``,
  (b) ``build_card()`` carries the CC BY 4.0 license, the Otake et al. 2024 citation, the
      paper DOI, and the LITERAL "intermediate validation build (v1.2)" note,
  (c) the private->public flip is GATED: ``make_public`` refuses (and never calls the HF
      API) unless ``confirm_public=True``; with the gate it calls
      ``update_repo_settings(private=False)`` on a MOCKED ``HfApi``,
  (d) ``HF_TOKEN`` is read from the environment only,
  (e) ``push_private`` preflights then pushes with ``private=True`` (mocked dataset),
  (f) ``smoke_load`` raises a clear ``RuntimeError`` (not a bare ``StopIteration``) on an
      empty stream,
  (g) the CLI's ``--publish`` public flip requires BOTH an explicit ``--public`` intent AND
      ``--confirm-public``: ``--publish --confirm-public`` alone never goes public.

The tests never touch the Hub: the only hub objects exercised are ``DatasetCard`` (built
offline from committed constants) and a fake ``HfApi`` / fake dataset.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from typing import ClassVar

import huggingface_hub
import pytest

from planktonzilla.dataset_import import frepj_layout
from planktonzilla.planktonzilla_dataset import frepj_publish as fp


class _FakeApi:
    """Records ``update_repo_settings`` kwargs instead of hitting the Hub."""

    calls: ClassVar[list[dict]] = []

    def update_repo_settings(self, **kwargs):
        _FakeApi.calls.append(kwargs)


class _FakeDataset:
    """Records ``push_to_hub`` kwargs instead of hitting the Hub."""

    def __init__(self):
        self.pushes: list[dict] = []

    def push_to_hub(self, repo_id, private=None, token=None):
        self.pushes.append({"repo_id": repo_id, "private": private, "token": token})


# --- (a) preflight allowlist + frozen-id guard ----------------------------------------


def test_preflight_rejects_frozen_full_repo_id():
    """preflight raises on the frozen full owner/name id."""
    with pytest.raises(ValueError):
        fp.preflight("project-oceania/planktonzilla-17M")


def test_preflight_rejects_frozen_bare_basename():
    """preflight raises on the bare frozen basename."""
    with pytest.raises(ValueError):
        fp.preflight("planktonzilla-17M")


def test_preflight_rejects_non_target_repo():
    """preflight raises on any non-frozen repo that is not the allowlisted target."""
    with pytest.raises(ValueError):
        fp.preflight("project-oceania/some-other-dataset")


def test_preflight_allows_frepj_target():
    """preflight returns None for the intended frepj target only."""
    assert fp.preflight("project-oceania/planktonzilla-frepj") is None
    assert fp.TARGET_REPO_ID == "project-oceania/planktonzilla-frepj"


# --- (b) dataset card content ---------------------------------------------------------


def test_build_card_carries_license_citation_and_literal_note():
    """build_card() text carries CC BY 4.0, Otake, the paper DOI, and the literal note."""
    content = fp.build_card().content
    assert "cc-by-4.0" in content
    assert "CC BY 4.0" in content
    assert "Otake" in content
    assert frepj_layout.PAPER_DOI in content
    assert frepj_layout.DATA_DOI in content
    # The EXACT phrasing distinguishing this build from 17M and the forthcoming v1.2.
    assert "intermediate validation build (v1.2)" in content
    assert fp.INTERMEDIATE_NOTE == "intermediate validation build (v1.2)"


# --- (c) gated public flip (mocked HfApi, no network) ---------------------------------


def test_make_public_refuses_without_confirm(monkeypatch):
    """make_public raises and NEVER calls the HF API unless confirm_public=True."""
    _FakeApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")

    with pytest.raises(ValueError):
        fp.make_public("project-oceania/planktonzilla-frepj", confirm_public=False)
    assert _FakeApi.calls == []  # the settings call was never reached


def test_make_public_flips_private_false_with_confirm(monkeypatch):
    """With confirm_public=True, make_public calls update_repo_settings(private=False) on a mocked HfApi."""
    _FakeApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")

    fp.make_public("project-oceania/planktonzilla-frepj", confirm_public=True)

    assert len(_FakeApi.calls) == 1
    call = _FakeApi.calls[0]
    assert call["repo_id"] == "project-oceania/planktonzilla-frepj"
    assert call["repo_type"] == "dataset"
    assert call["private"] is False


def test_make_public_rejects_frozen_even_with_confirm(monkeypatch):
    """The public flip still preflights: a frozen id is rejected even with --confirm-public."""
    _FakeApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")

    with pytest.raises(ValueError):
        fp.make_public("project-oceania/planktonzilla-17M", confirm_public=True)
    assert _FakeApi.calls == []


# --- (d) token read from env only -----------------------------------------------------


def test_resolve_token_reads_env(monkeypatch):
    """_resolve_token reads HF_TOKEN from the environment when no explicit token is passed."""
    monkeypatch.setenv("HF_TOKEN", "hf_fromenv")
    assert fp._resolve_token() == "hf_fromenv"


def test_resolve_token_raises_when_absent(monkeypatch):
    """_resolve_token raises when neither an explicit token nor HF_TOKEN is set."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError):
        fp._resolve_token()


# --- (e) push_private preflights and pushes PRIVATE (mocked dataset, no network) -------


def test_push_private_pushes_with_private_true(monkeypatch):
    """push_private preflights then pushes the mocked dataset with private=True."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    ds = _FakeDataset()
    fp.push_private(ds, "project-oceania/planktonzilla-frepj")

    assert len(ds.pushes) == 1
    push = ds.pushes[0]
    assert push["repo_id"] == "project-oceania/planktonzilla-frepj"
    assert push["private"] is True


def test_push_private_rejects_frozen_before_pushing(monkeypatch):
    """push_private rejects a frozen id BEFORE any push_to_hub call."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    ds = _FakeDataset()
    with pytest.raises(ValueError):
        fp.push_private(ds, "planktonzilla-17M")
    assert ds.pushes == []


# --- (f) smoke_load raises a clear RuntimeError on an empty stream (IN-01) -------------


def test_smoke_load_raises_clear_error_on_empty_stream(monkeypatch):
    """smoke_load raises a descriptive RuntimeError (not a bare StopIteration) on an empty stream."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: iter([]))

    with pytest.raises(RuntimeError, match="train split is empty"):
        fp.smoke_load("project-oceania/planktonzilla-frepj")


# --- CLI wiring: the public flip is gated behind --confirm-public ---------------------


def test_cli_defines_confirm_public_flag():
    """The CLI wires an explicit --confirm-public gate for any public flip."""
    parser = fp._build_parser()
    options = {action.dest for action in parser._actions}
    assert "confirm_public" in options
    assert "make_public" in options
    assert "card_only" in options
    assert "dataset_path" in options
    assert "repo_id" in options


def test_cli_public_without_confirm_errors():
    """--public without --confirm-public exits with a parser error (never auto-public)."""
    with pytest.raises(SystemExit):
        fp.main(["--public"])


def test_cli_make_public_without_confirm_refuses(monkeypatch):
    """--make-public without --confirm-public refuses at the make_public gate (no HF API call)."""
    _FakeApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    with pytest.raises(ValueError):
        fp.main(["--make-public"])
    assert _FakeApi.calls == []


# --- WR-02: --publish's public flip requires BOTH --public AND --confirm-public --------


def test_cli_publish_confirm_public_without_public_errors(monkeypatch):
    """--publish --confirm-public (no --public) is rejected by the parser, never reaches publish."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    ds = _FakeDataset()
    monkeypatch.setattr(fp, "load_built", lambda path: ds)
    monkeypatch.setattr(fp, "push_card", lambda repo_id, token=None: None)
    make_public_calls: list[tuple] = []
    monkeypatch.setattr(fp, "make_public", lambda *a, **k: make_public_calls.append((a, k)))

    with pytest.raises(SystemExit):
        fp.main(["--publish", "--confirm-public"])

    # Rejected before any push/card/make_public call — never went public, never pushed at all.
    assert ds.pushes == []
    assert make_public_calls == []


def test_cli_publish_alone_never_flips_public(monkeypatch):
    """Plain --publish (no --public, no --confirm-public) pushes privately and never calls make_public."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    ds = _FakeDataset()
    monkeypatch.setattr(fp, "load_built", lambda path: ds)
    monkeypatch.setattr(fp, "push_card", lambda repo_id, token=None: None)
    make_public_calls: list[tuple] = []
    monkeypatch.setattr(fp, "make_public", lambda *a, **k: make_public_calls.append((a, k)))

    fp.main(["--publish"])

    assert len(ds.pushes) == 1
    assert ds.pushes[0]["private"] is True
    assert make_public_calls == []


def test_cli_publish_public_and_confirm_public_flips_public(monkeypatch):
    """--publish --public --confirm-public (both explicit flags) DOES flip public."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    ds = _FakeDataset()
    monkeypatch.setattr(fp, "load_built", lambda path: ds)
    monkeypatch.setattr(fp, "push_card", lambda repo_id, token=None: None)
    make_public_calls: list[tuple] = []
    monkeypatch.setattr(fp, "make_public", lambda *a, **k: make_public_calls.append((a, k)))

    fp.main(["--publish", "--public", "--confirm-public"])

    assert len(ds.pushes) == 1
    assert ds.pushes[0]["private"] is False
    assert len(make_public_calls) == 1
    _, kwargs = make_public_calls[0]
    assert kwargs["confirm_public"] is True


# --- v1.2 schema: custom_metadata keys on smoke-load, columns on the card, --tag -------


def test_smoke_load_checks_custom_metadata_keys(monkeypatch):
    """The streamed example must carry the consolidated columns AND the FREPJ keys in custom_metadata."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    import datasets

    good = {column: "x" for column in fp.EXPECTED_FREPJ_COLUMNS}
    good["custom_metadata"] = '{"magnification": "40", "site": "biwako"}'
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: iter([good]))
    assert fp.smoke_load("project-oceania/planktonzilla-frepj") is True

    empty = dict(good, custom_metadata="{}")
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: iter([empty]))
    with pytest.raises(RuntimeError, match="lacks the FREPJ keys"):
        fp.smoke_load("project-oceania/planktonzilla-frepj")

    # The schema published on 2026-07-11 (top-level magnification/site/date, no license
    # columns, no custom_metadata) must FAIL the smoke: it is exactly what the republish replaces.
    legacy = {column: "x" for column in ("proposed_label", "magnification", "site", "date", "Latitude", "Longitude")}
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: iter([legacy]))
    with pytest.raises(RuntimeError, match="missing expected FREPJ columns"):
        fp.smoke_load("project-oceania/planktonzilla-frepj")


def test_build_card_documents_custom_metadata_and_timestamp():
    """The card tells a consumer where FREPJ's magnification/site/date went."""
    content = fp.build_card().content
    assert "## Columns" in content
    assert "custom_metadata" in content
    assert '"magnification": "40" | "100"' in content
    assert "`timestamp`" in content
    assert "license_url" in content


def test_cli_tag_calls_tag_release_with_default_and_explicit(monkeypatch):
    """--tag alone tags with DEFAULT_TAG; --tag NAME tags with NAME; both on the allowlisted repo."""
    calls = []
    monkeypatch.setattr(fp, "tag_release", lambda repo_id, tag: calls.append((repo_id, tag)))
    fp.main(["--tag"])
    fp.main(["--tag", "v9.9.9-frepj"])
    assert calls == [(fp.TARGET_REPO_ID, fp.DEFAULT_TAG), (fp.TARGET_REPO_ID, "v9.9.9-frepj")]


def test_tag_release_rejects_frozen_before_any_network_call(monkeypatch):
    """Tagging goes through the same preflight: the frozen 17M can never be tagged from here."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    with pytest.raises(ValueError, match="frozen"):
        fp.tag_release("project-oceania/planktonzilla-17M", "v1.2.0")


def test_build_card_carries_over_existing_metadata_but_owned_keys_win():
    """configs/dataset_info from the Hub card survive a card push; license/tags/pretty_name are ours."""
    existing = {
        "license": "mit",
        "tags": ["stale"],
        "configs": [{"config_name": "default", "data_files": [{"split": "train", "path": "data/train-*"}]}],
        "dataset_info": {
            "features": [{"name": "image", "dtype": "image"}],
            "splits": [{"name": "train", "num_examples": 88686}],
        },
    }
    data = fp.build_card(existing).data.to_dict()
    assert data["license"] == frepj_layout.LICENSE
    assert data["tags"] == ["plankton", "zooplankton", "image-classification", "frepj"]
    assert data["pretty_name"] == frepj_layout.HUMAN_READABLE_NAME
    assert data["configs"] == existing["configs"]
    assert data["dataset_info"] == existing["dataset_info"]
    # Offline default: no existing card -> exactly the owned keys.
    assert set(fp.build_card().data.to_dict()) == {"license", "tags", "pretty_name"}


def test_push_card_reads_the_existing_card_before_pushing(monkeypatch):
    """push_card loads the Hub card's metadata and hands it to build_card (no network here)."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    seen = {}
    monkeypatch.setattr(fp, "_existing_card_metadata", lambda repo_id, token: {"configs": ["kept"]})

    class _Card:
        def push_to_hub(self, repo_id, repo_type, token):
            seen["pushed"] = (repo_id, repo_type)

    def _build(existing=None):
        seen["existing"] = existing
        return _Card()

    monkeypatch.setattr(fp, "build_card", _build)
    fp.push_card("project-oceania/planktonzilla-frepj")
    assert seen == {"existing": {"configs": ["kept"]}, "pushed": ("project-oceania/planktonzilla-frepj", "dataset")}


def test_cli_smoke_hard_exits_after_the_check(monkeypatch):
    """--smoke ends with os._exit(0) once the check passed (interpreter shutdown otherwise hangs)."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    monkeypatch.setattr(fp, "smoke_load", lambda repo_id: True)
    exits = []
    monkeypatch.setattr(fp.os, "_exit", lambda code: exits.append(code))
    fp.main(["--smoke"])
    assert exits == [0]


def test_cli_smoke_failure_propagates_before_any_hard_exit(monkeypatch):
    """A failed smoke raises out of main() — the hard exit never masks a failure."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")

    def _fail(repo_id):
        raise RuntimeError("Smoke-load FAILED")

    monkeypatch.setattr(fp, "smoke_load", _fail)
    monkeypatch.setattr(fp.os, "_exit", lambda code: pytest.fail("os._exit must not run after a failed smoke"))
    with pytest.raises(RuntimeError, match="Smoke-load FAILED"):
        fp.main(["--smoke"])


def test_cli_other_steps_exit_normally(monkeypatch):
    """Only the smoke path hard-exits; a card push returns from main() like before."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    monkeypatch.setattr(fp, "push_card", lambda repo_id, token=None: None)
    monkeypatch.setattr(fp.os, "_exit", lambda code: pytest.fail("os._exit must not run for --push-card"))
    fp.main(["--push-card"])


def test_cli_configures_info_logging_when_nothing_is_configured(monkeypatch):
    """Run as a script, the helper's INFO verdicts must be visible; an existing setup is left alone."""
    import logging

    root_logger = logging.getLogger()
    saved = root_logger.handlers[:]
    monkeypatch.setattr(root_logger, "handlers", [])
    try:
        fp._ensure_console_logging()
        assert root_logger.handlers, "basicConfig must install a handler"
        assert root_logger.level == logging.INFO
        installed = root_logger.handlers[:]
        fp._ensure_console_logging()
        assert root_logger.handlers == installed, "second call must not add handlers"
    finally:
        for handler in root_logger.handlers:
            if handler not in saved:
                root_logger.removeHandler(handler)
        root_logger.handlers[:] = saved
