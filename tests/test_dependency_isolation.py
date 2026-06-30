"""
(c) Inria

Network-free dependency-isolation guard (Phase 9, FND-03 + FND-04).

The heavy viz stack (``gradio``, ``plotly``) must never leak into the frozen
training/dataset core. This guard enforces the boundary two ways, both pure
file/AST inspection (no ``import gradio`` / ``import plotly`` needed to RUN — so
it passes in the core env where neither is installed):

* Manifest assertion: ``gradio`` and ``plotly`` (exact PEP 508 names) are absent
  from ``[project.dependencies]`` and the ``dev`` group, and present ONLY in the
  ``explorer`` dependency group. Exact-name matching is used so the legitimate
  transitive ``gradio-client`` does NOT false-positive.
* AST module-scope scan: no ``*.py`` under ``planktonzilla/`` or ``tools/``
  imports ``gradio`` / ``plotly`` at module scope. Function/method-body imports
  (the lazy-import seam, e.g. ``tools/taxonomy_explorer.py``) are COMPLIANT.
  ``import gradio_client`` is a different top-level module and never triggers.

A negative leak-injection test proves the helpers actually FIRE on a simulated
leak (so the guard would go red on a real one).
"""

import ast
import re
import tomllib
from pathlib import Path

# Repo root resolved from this file's location (tests/ -> repo root), never cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# The viz distributions / top-level modules forbidden in core.
# kaleido (Phase 11, SANKEY-04 PNG-export backend) joins the boundary: it is added to
# the ISOLATED explorer group ONLY and is imported function-locally in
# planktonzilla/explorer/sankey.py (D4), so BOTH the manifest assertion (absent from core
# + dev, present only in explorer) AND the AST module-scope scan now cover it.
FORBIDDEN_NAMES = frozenset({"gradio", "plotly", "kaleido"})

# Core packages whose import-time graph must stay viz-free.
CORE_PACKAGE_DIRS = ("planktonzilla", "tools")

# PEP 508: a name ends at the first version operator, marker, or extras bracket.
_PEP508_NAME_RE = re.compile(r"[<>=!~;\[\s@]")


def _pkg_names(req_list: list[str]) -> set[str]:
    """Extract the set of canonical PEP 508 package names from a requirement list.

    Names are lowercased and hyphen-normalized (PEP 503), so ``gradio-client``
    stays distinct from ``gradio`` and a substring trap is impossible.
    """
    names: set[str] = set()
    for req in req_list:
        token = _PEP508_NAME_RE.split(req.strip(), 1)[0].strip()
        if token:
            names.add(token.lower().replace("_", "-"))
    return names


def _module_scope_viz_imports(source: str) -> list[str]:
    """Return module-scope imports of forbidden viz modules in ``source``.

    Walks ONLY the module body's top-level statements (and the bodies of
    top-level ``if``/``try`` blocks, which execute at import time). Imports
    nested inside ``FunctionDef``/``AsyncFunctionDef``/``ClassDef`` bodies are
    function-local (the lazy seam) and are NOT reported.

    Matches on the IMPORTED module's top-level token: ``import plotly.graph_objects``
    and ``from gradio import x`` are violations; ``import gradio_client`` is not
    (different top-level module name).
    """
    offending: list[str] = []
    tree = ast.parse(source)

    def _top_module(name: str) -> str:
        return name.split(".", 1)[0]

    def _scan(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                offending.extend(alias.name for alias in node.names if _top_module(alias.name) in FORBIDDEN_NAMES)
            elif isinstance(node, ast.ImportFrom):
                # `from . import x` has module=None; skip relative imports.
                if node.module and _top_module(node.module) in FORBIDDEN_NAMES:
                    offending.append(node.module)
            elif isinstance(node, ast.If):
                _scan(node.body)
                _scan(node.orelse)
            elif isinstance(node, ast.Try):
                _scan(node.body)
                _scan(node.handlers)  # type: ignore[arg-type]
                _scan(node.orelse)
                _scan(node.finalbody)
            elif isinstance(node, ast.ExceptHandler):
                _scan(node.body)
            elif isinstance(node, ast.With):
                _scan(node.body)
            # FunctionDef / AsyncFunctionDef / ClassDef bodies are NOT descended:
            # function-local imports are the compliant lazy-import seam.

    _scan(tree.body)
    return offending


def _load_manifest() -> dict:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def test_manifest_isolates_viz_deps() -> None:
    """gradio/plotly live ONLY in the explorer group, never in core or dev."""
    manifest = _load_manifest()
    core = _pkg_names(manifest["project"]["dependencies"])
    groups = manifest["dependency-groups"]
    dev = _pkg_names(groups.get("dev", []))
    explorer = _pkg_names(groups.get("explorer", []))

    for name in FORBIDDEN_NAMES:
        assert name not in core, f"{name} leaked into [project.dependencies]"
        assert name not in dev, f"{name} leaked into the dev group"
        assert name in explorer, f"{name} missing from the explorer group"

    # The substring trap: gradio-client is a legitimate transitive and is NOT
    # the same distribution as gradio. It must not be confused with it.
    assert "gradio-client" not in core
    assert "gradio-client" not in explorer  # only direct gradio is pinned here


def test_no_module_scope_viz_imports_in_core() -> None:
    """No core package imports gradio/plotly at module scope (lazy seam only)."""
    violations: dict[str, list[str]] = {}
    for pkg in CORE_PACKAGE_DIRS:
        pkg_dir = _REPO_ROOT / pkg
        if not pkg_dir.exists():
            continue
        for py_file in sorted(pkg_dir.rglob("*.py")):
            offending = _module_scope_viz_imports(py_file.read_text(encoding="utf-8"))
            if offending:
                violations[str(py_file.relative_to(_REPO_ROOT))] = offending

    assert not violations, f"module-scope viz imports found in core packages: {violations}"


def test_guard_fires_on_injected_leak() -> None:
    """Negative case: the helpers MUST flag a simulated leak (guard goes red)."""
    # Manifest helper flags gradio when it is (wrongly) in core dependencies.
    leaked_core = _pkg_names(["torch>=2.0", "gradio>=6.0,<7", "polars"])
    assert "gradio" in leaked_core, "manifest helper failed to detect leaked gradio"

    # AST helper flags a module-scope `import plotly` and `from gradio import X`.
    leaked_source = "import os\nimport plotly.graph_objects as go\nfrom gradio import Blocks\n"
    offending = _module_scope_viz_imports(leaked_source)
    assert "plotly.graph_objects" in offending
    assert "gradio" in offending

    # And it must NOT fire on the compliant function-local seam ...
    compliant_local = "def build():\n    import gradio as gr\n    import plotly.graph_objects as go\n    return gr, go\n"
    assert _module_scope_viz_imports(compliant_local) == []

    # ... nor on the gradio-client transitive (different top-level module).
    compliant_client = "import gradio_client\nfrom gradio_client import Client\n"
    assert _module_scope_viz_imports(compliant_client) == []
    assert "gradio" not in _pkg_names(["gradio-client>=1.0"])
