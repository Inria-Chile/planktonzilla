"""
(c) Inria

OmegaConf resolver definitions for planktonzilla.

Importing this module is the single side-effect that registers all
custom OmegaConf resolvers.  Both entry-point modules (``train.py`` and
``import_dataset.py``) import this module explicitly so that resolvers
are available before Hydra composes any config.

The registration call is guarded by ``try/except ValueError: pass`` because
``OmegaConf.register_new_resolver`` raises ``ValueError`` on a duplicate name.
Tolerating that lets the module be re-imported (e.g. by both entry points, or
across test runs in one process) without crashing on the already-registered
resolver, while keeping registration idempotent.
"""

from omegaconf import OmegaConf


def strip_yaml_suffix(s: str) -> str:
    """Strip a single trailing ``.yaml`` suffix from ``s``; pass-through otherwise.

    Replaces the prior ``eval`` OmegaConf resolver (CONCERNS #7 RCE vector).
    Used by ``configs/experiment/*.yaml`` to derive bare names from
    ``${hydra:runtime.choices.<group>}`` which arrive with the ``.yaml`` suffix.
    """
    if s.endswith(".yaml"):
        return s[:-5]
    return s


try:
    OmegaConf.register_new_resolver("strip_yaml_suffix", strip_yaml_suffix)
except ValueError:
    pass
