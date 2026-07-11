"""
(c) Inria

Reusable frozen-repo preflight guard.

The frozen ``planktonzilla-17M`` dataset (and the models trained on it) are
published and immutable on the HuggingFace Hub. Any generation / publish flow that
could push to the Hub must first prove its target is NOT a frozen artifact. This
module derives the frozen repo-id set from ``constants`` (the single source of
truth — the literal id string is never re-typed here) and exposes
``assert_not_frozen_repo`` as the reusable preflight the Phase-19 publish helper
(Plan 19-02) and, later, Phase-20's PUB-03 guard import.
"""

from planktonzilla.planktonzilla_dataset import constants

# The intended intermediate-validation target (Plan 19). Kept as an explicit
# allowlist entry so a future typo that collides with a frozen id is caught early.
FREPJ_ONLY_REPO_ID = "project-oceania/planktonzilla-frepj"

# Frozen, immutable published artifacts — derived from constants so there is ONE
# source of truth. Both the full ``owner/name`` repo id and its bare basename are
# rejected (a bare "planktonzilla-17M" push target is just as dangerous).
FROZEN_REPO_IDS = frozenset(
    {
        constants.DEFAULT_PLANKTONZILLA_DATASET_REPO_ID,
        constants.DEFAULT_PLANKTONZILLA_DATASET_NAME,
    }
)


def _basename(repo_id: str) -> str:
    """Return the ``owner/name`` -> ``name`` basename (or the whole string)."""
    return repo_id.rsplit("/", 1)[-1]


def is_frozen_repo(repo_id: str) -> bool:
    """Return ``True`` when ``repo_id`` (or its basename) is a known frozen id.

    The basename comparison is case-insensitive as a defensive extra so a stray
    ``Planktonzilla-17M`` capitalisation cannot slip past the guard.
    """
    if repo_id in FROZEN_REPO_IDS:
        return True

    frozen_basenames = {_basename(frozen).lower() for frozen in FROZEN_REPO_IDS}
    return _basename(repo_id).lower() in frozen_basenames


def assert_not_frozen_repo(repo_id: str) -> None:
    """Raise ``ValueError`` if ``repo_id`` targets a frozen, published artifact.

    Reusable preflight for every Hub-push path. Returns ``None`` (the target is
    allowed) for any non-frozen repo id, e.g. the intermediate
    ``project-oceania/planktonzilla-frepj`` validation repo or a future
    ``planktonzilla-v1.2`` release.
    """
    if is_frozen_repo(repo_id):
        raise ValueError(
            f"Refusing to push to frozen repo «{repo_id}»: it matches a published, immutable "
            f"artifact ({sorted(FROZEN_REPO_IDS)}). The frozen planktonzilla-17M dataset must "
            "never be overwritten — push to an intermediate / versioned repo id instead."
        )
