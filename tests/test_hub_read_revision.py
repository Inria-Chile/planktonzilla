"""
(c) Inria

Structural guard: every path that READS the published dataset can be pinned to a Hub revision.

The write side has forwarded ``push_revision`` to ``push_to_hub`` since the licence columns
landed. The read side had no matching half, so a column published onto a branch was invisible to
every tool that reads the dataset back — and the documented additive re-push was therefore wrong
on its SECOND run: it re-read the default branch, found no new column, and concatenated a
null-filled one over what the first run had published. Nothing went red; the columns simply
emptied.

Three guards, in increasing order of what they would have caught:

* ``revision_kwargs`` forwards NOTHING when unset. That is not fussiness: ``revision=None`` is a
  different call to ``datasets`` with a different cache key, and it breaks the test doubles that
  pin a one- and a two-parameter signature elsewhere in this suite.
* Every inventoried read path ACCEPTS a revision, by signature. A path that loses the parameter
  in a refactor goes red here rather than silently reading the default branch again.
* No Hub read sits OUTSIDE the inventory. This is the one that matters: two of the eight readers
  are invisible to ``grep load_dataset`` — ``check_base_on_hub``'s ``dataset_info`` pre-flight,
  and an ``HfFileSystem`` glob inside a notebook. A source scan finds both, and parsing the
  notebook JSON is the only way the second is reachable at all.

Network-free: pure ``inspect`` and file/AST inspection, nothing imported that speaks HTTP.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import ast
import inspect
import json
from pathlib import Path

import pytest

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset import make_planktonzilla as mk
from planktonzilla.planktonzilla_dataset import sankey as sk

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Every function that reads the PUBLISHED artifact and must therefore accept a revision. Frozen
# by name: adding a reader means adding it here, which is the point.
_INVENTORY = (
    (mk.load_base, "location"),  # takes a BaseLocation, which carries .revision
    (mk.check_base_on_hub, "revision"),
    (sk.scan_dataset, "revision"),
    (sk.fetch_dataset_metadata, "revision"),
    (sk.provenance, "revision"),
)

# The modules the source scan covers, and the functions in each that are ALLOWED to speak to the
# Hub. Anything else calling one of the read builtins below is an un-inventoried reader.
_SCANNED = {
    "planktonzilla/planktonzilla_dataset/make_planktonzilla.py": {"load_base", "check_base_on_hub", "check_push_target"},
    "planktonzilla/planktonzilla_dataset/sankey.py": {"scan_dataset", "fetch_dataset_metadata"},
    "planktonzilla/planktonzilla_dataset/gen_planktonzilla_only_plankton.py": {"main"},
    "planktonzilla/planktonzilla_dataset/update_planktonzilla.py": {"main"},
}

# The calls that reach the Hub. `HfApi()` itself is inert; `dataset_info` is the read.
_HUB_READS = ("load_dataset", "dataset_info", "HfFileSystem", "list_repo_files", "snapshot_download")


def test_revision_kwargs_is_empty_unless_the_revision_is_set():
    """Unset means unchanged — forwarding nothing, not `revision=None`.

    The empty dict is what keeps every default run byte-identical and what lets the existing
    one- and two-parameter test doubles stay unmodified. Falsy values are all treated as unset,
    because an empty string from a shell override is "the user did not pass one".
    """
    for unset in (None, "", 0, False):
        assert constants.revision_kwargs(unset) == {}

    assert constants.revision_kwargs("v1.1") == {"revision": "v1.1"}
    assert constants.revision_kwargs("main") == {"revision": "main"}

    # A fresh dict per call, so a caller cannot mutate the next caller's kwargs.
    first = constants.revision_kwargs("v1.1")
    first["revision"] = "tampered"
    assert constants.revision_kwargs("v1.1") == {"revision": "v1.1"}


@pytest.mark.parametrize(("function", "parameter"), _INVENTORY, ids=lambda x: getattr(x, "__name__", x))
def test_every_published_read_path_accepts_a_revision(function, parameter):
    """By signature, so a refactor that drops the parameter goes red here."""
    assert parameter in inspect.signature(function).parameters, (
        f"{function.__module__}.{function.__qualname__} no longer accepts «{parameter}», so the "
        f"read it performs can only ever see the repo's default branch."
    )


def test_sankey_exposes_a_revision_flag_distinct_from_its_version_flag():
    """Sankey is the CLI reader whose parser is inspectable, so assert against the parser."""
    sankey_flags = {action.option_strings[0] for action in sk._build_parser()._actions if action.option_strings}
    assert "--dataset-revision" in sankey_flags

    # `--dataset-revision` is deliberately NOT `--dataset-version`: one pins what is READ, the
    # other pins the string the page prints and reads nothing. Both must exist, distinctly.
    assert "--dataset-version" in sankey_flags


def test_the_only_plankton_builder_exposes_a_revision_flag():
    """Its parser is built inline in `main`, so the flag is asserted against the source."""
    module = _REPO_ROOT / "planktonzilla/planktonzilla_dataset/gen_planktonzilla_only_plankton.py"
    source = module.read_text(encoding="utf-8")
    assert '"--revision"' in source
    assert "revision_kwargs(args.revision)" in source


def test_the_hydra_readers_declare_a_null_defaulted_revision_key():
    """Null by default, or the pin would change what today's documented commands do."""
    for config, key in (
        ("configs/planktonzilla.yaml", "base_revision"),
        ("configs/update_planktonzilla.yaml", "read_revision"),
    ):
        text = (_REPO_ROOT / config).read_text(encoding="utf-8")
        assert f"\n{key}: null\n" in text, f"{config} must declare `{key}: null`"


def test_no_hub_read_sits_outside_an_inventoried_function():
    """The guard that would have caught the pre-flight and the notebook.

    `check_base_on_hub` reads the Hub through `dataset_info`, not `load_dataset`, and the
    sampling-map notebook reads it through an `HfFileSystem` glob. Neither is findable by
    grepping for the obvious call, and both were unpinned until this landed.
    """
    offenders = []

    for relative, allowed in _SCANNED.items():
        path = _REPO_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in _HUB_READS:
                continue
            enclosing = _enclosing_function(tree, node)
            if enclosing not in allowed:
                offenders.append(f"{relative}:{node.lineno} — {name}() inside {enclosing or '<module>'}")

    assert not offenders, "Hub read(s) outside the inventory:\n  " + "\n  ".join(offenders)


def test_the_sampling_notebook_pins_its_revision_and_keys_its_cache_on_it():
    """The eighth reader, and the only one a Python source scan cannot see.

    Two separate defects lived here. The glob was unpinned, and `CACHE_PATH` was keyed on
    nothing — so a cache built from the default branch survived a switch to a branch and was
    silently reused, and the notebook drew one revision's figure from another revision's bytes.
    """
    notebook = _REPO_ROOT / "notebooks/02. Sampling_world_map.ipynb"
    source = "\n".join("".join(cell["source"]) for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"])

    assert "REVISION = None" in source, "the notebook must declare a revision knob, defaulting to None"
    assert "{repo_id}@{revision}" in source, "the glob must be able to name a revision"
    assert "REVISION or 'main'" in source, "the parquet cache filename must be keyed on the revision"


def _called_name(node):
    """`load_dataset(...)` -> "load_dataset"; `api.dataset_info(...)` -> "dataset_info"."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _enclosing_function(tree, target):
    """The name of the nearest enclosing def, or None at module scope.

    Nested defs resolve to the OUTERMOST one, because `read_shard` inside `scan_dataset` is part
    of that function's Hub read rather than a reader of its own.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(child is target for child in ast.walk(node)):
            return node.name
    return None


def test_the_scan_fires_on_an_injected_unpinned_reader(tmp_path):
    """A guard that cannot go red is indistinguishable from no guard.

    The scan above passes today, which proves nothing on its own — it would also pass if
    `_called_name` returned None for every node, or if `_enclosing_function` never resolved a
    name. This injects the exact defect the guard exists to catch, in both shapes that defeated
    a grep, and asserts it is found and attributed to the right function.
    """
    injected = tmp_path / "leaky.py"
    injected.write_text(
        "def innocent():\n"
        "    return 1\n"
        "\n"
        "def sneaky_loader():\n"
        '    return load_dataset("org/ds", split="train")\n'
        "\n"
        "def sneaky_preflight(api):\n"
        '    return api.dataset_info("org/ds")\n',
        encoding="utf-8",
    )

    tree = ast.parse(injected.read_text(encoding="utf-8"))
    found = {
        (_enclosing_function(tree, node), _called_name(node))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) in _HUB_READS
    }

    assert found == {("sneaky_loader", "load_dataset"), ("sneaky_preflight", "dataset_info")}
    assert all(name is not None for name, _ in found), "the enclosing function must be attributed"
