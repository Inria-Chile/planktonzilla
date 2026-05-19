"""
(c) Inria
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

import planktonzilla.train  # noqa: F401  --- import-time side effect registers resolvers


def test_eval_resolver_removed():
    """Pin that ${eval:...} is no longer a registered OmegaConf resolver.

    The prior implementation registered `eval` as a resolver, which was an
    arbitrary-code-execution vector via configs and CLI overrides. After
    FOCUS-02 the `eval` registration is gone; only the narrow
    `strip_yaml_suffix` resolver remains. Closes CONCERNS.md #7.
    """
    cfg = OmegaConf.create({"x": "${eval:'1+1'}"})
    with pytest.raises(Exception) as exc_info:
        _ = cfg.x
    msg = str(exc_info.value).lower()
    assert "eval" in msg or "resolver" in msg, f"unexpected error message: {exc_info.value}"


def test_strip_yaml_suffix():
    """Pin the narrow strip_yaml_suffix resolver behavior.

    Strips a single trailing ``.yaml`` suffix; pass-through otherwise.
    Replaces the prior ``eval``-based ``[0:-5]`` slicing trick used in
    ``configs/experiment/*.yaml``.
    """
    assert OmegaConf.create({"x": "${strip_yaml_suffix:foo.yaml}"}).x == "foo"
    assert OmegaConf.create({"x": "${strip_yaml_suffix:nosuffix}"}).x == "nosuffix"
    assert OmegaConf.create({"x": "${strip_yaml_suffix:multi.yaml.yaml}"}).x == "multi.yaml"
