"""
(c) Inria

Guard: the linter CI runs is the linter a contributor runs.

`ruff check` / `ruff format --check` gate every push, so which ruff executes decides
whether a diff lands. It used to be decided by the calendar instead. ruff appeared in
neither `[project.dependencies]` nor `[dependency-groups]` nor `uv.lock`, so:

  * a contributor's `uv run ruff` resolved it ephemerally — whatever was newest then;
  * CI's own `uv tool install ruff` resolved it again, separately — whatever was newest
    when the job ran.

Two unpinned resolutions minutes or weeks apart. That is not hypothetical: `FURB192` is a
*preview* rule in 0.15.8 and stable in a later release, so a tree that was clean locally
failed lint here on a diff that had nothing to do with it. The reverse is worse — a real
finding invisible locally until an unrelated PR happens to pick up a newer ruff.

The fix is one source of truth: ruff is a `dev` dependency pinned to an exact version in
`pyproject.toml`, and CI does nothing but `uv sync`.

The pin has to live in `pyproject.toml` specifically. `uv.lock` is gitignored here
(`.gitignore:279`), so it never reaches CI — a `>=` floor plus a lockfile CI cannot see
resolves to "newest at job time", which is the very thing being fixed. Hence `==`, in the
one file both sides actually read.

Network-free: reads `pyproject.toml`, `.github/workflows/*.yml` and the installed
distribution metadata. Nothing is executed.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import importlib.metadata
import re
import tomllib

import pytest

WORKFLOWS = sorted((root / ".github" / "workflows").glob("*.yml"))

# An out-of-band install re-resolves the version and defeats the lockfile, whichever
# installer does it.
_OUT_OF_BAND = re.compile(r"(uv tool install|uvx|pip install|pipx install|uv pip install)[^\n]*\bruff\b")


def _dev_group():
    manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return manifest["dependency-groups"]["dev"]


def _pinned_ruff_version():
    """The exact version `pyproject.toml` pins, or ``None`` if it does not pin one."""
    for spec in _dev_group():
        match = re.fullmatch(r"ruff==(\d+\.\d+\.\d+)", spec.strip())
        if match:
            return match.group(1)
    return None


def _out_of_band_installs(text: str):
    """Lines that install ruff outside the lockfile, ignoring comments.

    Comment text is stripped first, in YAML and in the shell alike — otherwise this file's
    own explanation of why the workflow must NOT do this reads as an instance of it.
    """
    offenders = []
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        if _OUT_OF_BAND.search(code):
            offenders.append(line.strip())
    return offenders


# ── The manifest ───────────────────────────────────────────────────────────────────────────
def test_ruff_is_declared_as_a_dependency():
    """Not declared means `uv sync` does not install it, and `uv run` invents a version."""
    assert any(spec.split()[0].startswith("ruff") for spec in _dev_group()), (
        "ruff is absent from [dependency-groups].dev — CI and contributors will each resolve "
        "their own copy, which is the drift this file exists against"
    )


# ── The pin ────────────────────────────────────────────────────────────────────────────────
def test_the_declared_ruff_version_is_exact():
    """A `>=` floor is not a pin when the lockfile is gitignored — CI would still drift.

    This is the trap the first attempt at this fix fell into: `ruff>=…` plus `uv lock` looks
    pinned locally and is not pinned at all in CI, which never sees `uv.lock`.
    """
    assert _pinned_ruff_version(), (
        "ruff is not pinned to an exact version in [dependency-groups].dev. `uv.lock` is "
        "gitignored, so CI never sees it — the `==` in pyproject.toml is the only pin there is."
    )


def test_the_installed_ruff_is_the_one_that_is_pinned():
    """The strongest form: what runs here equals what CI will run.

    A mismatch means this environment is stale, so a local `ruff check` says nothing about
    what the push will do. `uv sync` fixes it.
    """
    assert importlib.metadata.version("ruff") == _pinned_ruff_version(), (
        "the installed ruff differs from the pinned one — run `uv sync` before trusting a local lint"
    )


# ── The workflows ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_no_workflow_installs_ruff_out_of_band(workflow):
    """CI must take ruff from `uv sync`, never re-resolve it."""
    offenders = _out_of_band_installs(workflow.read_text(encoding="utf-8"))
    assert not offenders, f"{workflow.name} installs ruff outside the lockfile: {offenders}"


def test_the_workflow_glob_is_not_silently_empty():
    """Guards the parametrisation: no workflows would make the suite above vacuous."""
    assert WORKFLOWS, "no workflow files found — the path moved and this guard stopped guarding"


def test_the_scanner_fires_on_a_simulated_out_of_band_install():
    """Negative injection, so a green result above means the scanner still works."""
    assert _out_of_band_installs("      - run: uv tool install ruff\n") == ["- run: uv tool install ruff"]
    assert _out_of_band_installs("      - run: pip install ruff==0.16.5\n")
    assert _out_of_band_installs("      - run: uvx ruff check .\n")
    # ...and stays quiet on the shape the workflow actually uses.
    assert _out_of_band_installs("      - run: uv run ruff check planktonzilla/ tests/\n") == []
    assert _out_of_band_installs("      - run: uv sync\n") == []
    # ...and on prose about the thing, including this repo's own comment saying not to do it.
    assert _out_of_band_installs("      # No `uv tool install ruff` here on purpose.\n") == []
    assert _out_of_band_installs("      - run: uv sync  # not `uv tool install ruff`\n") == []
