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


def _normalize(repo_id: str) -> str:
    """Strip surrounding whitespace and a trailing ``/`` before any comparison.

    A trailing slash makes ``rsplit("/", 1)[-1]`` return an empty string (matching
    nothing), and stray whitespace adjacent to the last ``/`` becomes part of the
    basename and breaks the exact-match comparison. Normalizing first closes both
    bypasses without changing behavior for any already-well-formed repo id.
    """
    return repo_id.strip().rstrip("/")


def _basename(repo_id: str) -> str:
    """Return the ``owner/name`` -> ``name`` basename (or the whole string).

    The extracted basename is also stripped so a stray space adjacent to the
    ``/`` separator (e.g. ``"project-oceania/ planktonzilla-17M"``) does not
    survive into the basename comparison.
    """
    return _normalize(repo_id).rsplit("/", 1)[-1].strip()


def is_frozen_repo(repo_id: str) -> bool:
    """Return ``True`` when ``repo_id`` (or its basename) is a known frozen id.

    ``repo_id`` is normalized first (surrounding whitespace and a trailing ``/``
    stripped) so trivial, plausible variants of the frozen id — a trailing slash
    from a copy-pasted Hydra override, a stray leading/trailing space from shell
    quoting — cannot evade detection. The basename comparison is also
    case-insensitive as a defensive extra so a stray ``Planktonzilla-17M``
    capitalisation cannot slip past the guard.
    """
    normalized = _normalize(repo_id)
    if normalized in FROZEN_REPO_IDS:
        return True

    frozen_basenames = {_basename(frozen).lower() for frozen in FROZEN_REPO_IDS}
    return _basename(normalized).lower() in frozen_basenames


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
