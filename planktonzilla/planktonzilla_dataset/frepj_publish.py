"""
(c) Inria

Frozen-repo-guarded publish helper for the intermediate FREPJ-only validation dataset
(Plan 19-02, VAL-01).

ORCHESTRATOR-RUN. This module AUTHORS the publish mechanism; it is NEVER auto-run by a
backgrounded agent. The publish is staged so the orchestrator can run each step visibly:

    load-from-disk -> preflight guard -> push PRIVATE -> smoke-load -> gated PUBLIC flip + card

Safety design (mirrors the proven Phase-14 ``deploy/deploy_space.py`` shape, for a
``repo_type="dataset"`` repo):

- **Allowlisted target.** The ONLY permitted push target is
  ``project-oceania/planktonzilla-frepj``. :func:`preflight` runs before EVERY
  push/create/settings call and (a) rejects any frozen artifact id via
  :func:`planktonzilla.planktonzilla_dataset.frozen_repo_guard.assert_not_frozen_repo`
  and (b) asserts ``repo_id == TARGET_REPO_ID`` as a belt-and-suspenders allowlist. The
  frozen ``planktonzilla-17M`` dataset can therefore never be overwritten.
- **Private-first.** The dataset is pushed PRIVATE, smoke-loaded, and only flipped PUBLIC
  behind an explicit ``--confirm-public`` gate. The public flip never happens by default.
- **Secret hygiene.** ``HF_TOKEN`` is read from the environment only; it is never
  hardcoded, printed, or embedded in the card or data.

Procedure (orchestrator runs these in order, with ``HF_TOKEN`` set, write scope on
project-oceania — see Plan 19-02 Task 2):

    uv run python -m planktonzilla.planktonzilla_dataset.frepj_publish --push-private
    uv run python -m planktonzilla.planktonzilla_dataset.frepj_publish --smoke
    uv run python -m planktonzilla.planktonzilla_dataset.frepj_publish --push-card
    uv run python -m planktonzilla.planktonzilla_dataset.frepj_publish --make-public --confirm-public
    uv run python -m planktonzilla.planktonzilla_dataset.frepj_publish --tag v1.2.0-frepj

Zero behavioral drift: nothing here touches, downloads, or mutates any frozen artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from planktonzilla.dataset_import import frepj_layout
from planktonzilla.planktonzilla_dataset import constants, frozen_repo_guard
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# The ONLY allowlisted push target (belt-and-suspenders on top of the frozen-id guard).
# Aliased from frozen_repo_guard.FREPJ_ONLY_REPO_ID so there is ONE source of truth for
# this security-relevant literal — it is never re-typed here.
TARGET_REPO_ID = frozen_repo_guard.FREPJ_ONLY_REPO_ID

# Default location of the locally-built + validated FREPJ-only dataset (Plan 19-01).
DEFAULT_DATASET_PATH = "data/frepj_only_build/planktonzilla-17M"

# Verbatim note (from 19-CONTEXT) distinguishing this build from the frozen 17M composite
# and the forthcoming full ``planktonzilla-v1.2``. Kept as a constant so the tests and the
# card share ONE source of truth for the exact phrasing.
INTERMEDIATE_NOTE = "intermediate validation build (v1.2)"

# Columns the smoke-load must find on a streamed example to call the load a PASS, and the
# keys its `custom_metadata` JSON object must carry. FREPJ has no column of its own: the
# magnification and the raw site token live in `custom_metadata`, the normalized
# sampling date in `timestamp` (see FrepjRedefiner).
EXPECTED_FREPJ_COLUMNS = (
    "proposed_label",
    "timestamp",
    "Latitude",
    "Longitude",
    constants.CUSTOM_METADATA_COL,
    *constants.LICENSE_COLS,
)
EXPECTED_CUSTOM_METADATA_KEYS = ("magnification", "site")

# Default Hub tag for a republish of this intermediate build (`--tag` with no value).
DEFAULT_TAG = "v1.2.0-frepj"

# Bounded push retry budget (mirrors DatasetImporter._push_to_hub).
PUSH_RETRIES = 10


def preflight(repo_id: str) -> None:
    """Assert ``repo_id`` is a legal, non-frozen push target — raise ``ValueError`` otherwise.

    Runs before EVERY hub operation (push / card / settings). Two layered checks:

    1. :func:`frozen_repo_guard.assert_not_frozen_repo` — rejects any known frozen id
       (full ``owner/name`` or bare basename, case-insensitive).
    2. An allowlist assertion that ``repo_id == TARGET_REPO_ID`` — belt-and-suspenders so a
       typo that is *not* a frozen id still cannot be published to by accident.

    Returns ``None`` when ``repo_id`` is exactly the intended
    ``project-oceania/planktonzilla-frepj`` target.
    """
    frozen_repo_guard.assert_not_frozen_repo(repo_id)
    if repo_id != TARGET_REPO_ID:
        raise ValueError(
            f"Refusing to publish to «{repo_id}»: the only allowlisted FREPJ-only target is "
            f"«{TARGET_REPO_ID}». Publish there, or update TARGET_REPO_ID deliberately."
        )


def _resolve_token(hf_token: str | None = None) -> str:
    """Return the HF token from the argument or the ``HF_TOKEN`` env var; never printed.

    Raises ``ValueError`` when neither is set — the token is read from the environment
    only, never hardcoded.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN is not set: export HF_TOKEN (write scope on project-oceania) before publishing. "
            "The token is read from the environment only and is never printed or committed."
        )
    return token


def load_built(path: str = DEFAULT_DATASET_PATH):
    """Load the locally-built FREPJ-only dataset saved with ``save_to_disk`` (Plan 19-01)."""
    import datasets

    logger.info(f"Loading the built FREPJ-only dataset from «{path}».")
    return datasets.load_from_disk(path)


def _push_dataset(ds, repo_id: str, token: str, private: bool, retries: int = PUSH_RETRIES) -> None:
    """Push ``ds`` to ``repo_id`` with a bounded retry loop (mirrors ``_push_to_hub``).

    Re-raises the last exception as ``RuntimeError`` once the retry budget is exhausted so
    the caller gets a clear pass/fail rather than a silent skip.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            ds.push_to_hub(repo_id, private=private, token=token)
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(f"Push attempt {attempt + 1}/{retries} to «{repo_id}» failed, retrying. Cause: {exc}.")
    raise RuntimeError(f"Failed to push «{repo_id}» after {retries} attempts.") from last_exc


def push_private(ds, repo_id: str = TARGET_REPO_ID, token: str | None = None, retries: int = PUSH_RETRIES) -> None:
    """Preflight, then push ``ds`` to ``repo_id`` as PRIVATE (creates the repo private on first push)."""
    preflight(repo_id)
    token = _resolve_token(token)
    logger.info(f"Pushing the FREPJ-only dataset to «{repo_id}» as PRIVATE.")
    _push_dataset(ds, repo_id, token, private=True, retries=retries)


def smoke_load(repo_id: str = TARGET_REPO_ID, token: str | None = None) -> bool:
    """Stream one example from ``repo_id`` and assert the expected FREPJ columns are present.

    Returns ``True`` on a clean load; raises ``RuntimeError`` when any expected FREPJ column
    is missing. Runs :func:`preflight` first (belt-and-suspenders even for a read).
    """
    preflight(repo_id)
    token = _resolve_token(token)
    import datasets

    logger.info(f"Smoke-loading «{repo_id}» (streaming) to verify the FREPJ columns.")
    stream = datasets.load_dataset(repo_id, split="train", streaming=True, token=token)
    try:
        example = next(iter(stream))
    except StopIteration:
        raise RuntimeError(f"Smoke-load FAILED: «{repo_id}» train split is empty.") from None
    missing = [column for column in EXPECTED_FREPJ_COLUMNS if column not in example]
    if missing:
        raise RuntimeError(f"Smoke-load FAILED: «{repo_id}» is missing expected FREPJ columns {missing}.")
    try:
        custom = json.loads(example[constants.CUSTOM_METADATA_COL] or "{}")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Smoke-load FAILED: «{repo_id}» {constants.CUSTOM_METADATA_COL} is not JSON: {exc}.") from exc
    missing_keys = [key for key in EXPECTED_CUSTOM_METADATA_KEYS if key not in custom]
    if missing_keys:
        raise RuntimeError(
            f"Smoke-load FAILED: «{repo_id}» {constants.CUSTOM_METADATA_COL} lacks the FREPJ keys {missing_keys}."
        )
    logger.info(
        f"Smoke-load PASS: «{repo_id}» carries all expected FREPJ columns {list(EXPECTED_FREPJ_COLUMNS)} "
        f"and {constants.CUSTOM_METADATA_COL} keys {list(EXPECTED_CUSTOM_METADATA_KEYS)}."
    )
    return True


# The card metadata this helper owns. Anything ELSE found in the card already on the Hub
# is carried over untouched — in particular the `configs` / `dataset_info` blocks that
# `push_to_hub` writes (features, split sizes, data_files), which the dataset viewer and
# `load_dataset` read. Replacing the whole README, as the first publish did, silently
# dropped them.
_OWNED_CARD_METADATA = {
    "license": frepj_layout.LICENSE,
    "tags": ["plankton", "zooplankton", "image-classification", "frepj"],
    "pretty_name": frepj_layout.HUMAN_READABLE_NAME,
}


def _card_content(existing_metadata: dict | None = None) -> str:
    """Build the FREPJ dataset-card markdown (YAML header + body) from committed constants.

    REUSES the ``frepj_layout`` license/citation/DOI constants so the citation is never
    re-transcribed. The body carries FREPJ attribution, the CC BY 4.0 license, the Otake
    et al. 2024 citation (paper + data DOIs), and the LITERAL ``INTERMEDIATE_NOTE``
    distinguishing this build from the frozen ``planktonzilla-17M`` and the forthcoming
    full ``planktonzilla-v1.2``.

    ``existing_metadata`` is the YAML block of the card already on the Hub (if any): it
    is carried over key for key, with the owned keys (license, tags, pretty_name)
    overriding, so the ``configs`` / ``dataset_info`` that ``push_to_hub`` maintains
    survive a card push.
    """
    import yaml

    metadata = {**(existing_metadata or {}), **_OWNED_CARD_METADATA}
    header = "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + "---\n\n"
    body = f"""# {frepj_layout.HUMAN_READABLE_NAME}

**{INTERMEDIATE_NOTE}** — this repository is the FREPJ-only intermediate testing/validation
build. It is deliberately distinct from the frozen, immutable
`project-oceania/planktonzilla-17M` composite (which is NOT modified by this build) and from
the forthcoming full composite `planktonzilla-v1.2`.

## Attribution

FREPJ-Z (Freshwater Plankton in Japanese Lakes and Reservoirs, I. Zooplankton), sourced from
<{frepj_layout.SOURCE_URL}>.

## Columns

The consolidated planktonzilla schema — `image`, the source provenance (`dataset`,
`original_label`, `original_path`), the seven taxonomy ranks and label extras, the five
external-ID columns, the metadata columns (`Latitude`, `Longitude`, `timestamp`, …) and the
per-image `license` / `license_url` — plus, as in every planktonzilla source, one JSON object
per image in `{constants.CUSTOM_METADATA_COL}` holding what only this source knows:

- `{constants.CUSTOM_METADATA_COL}`: `{{"magnification": "40" | "100", "site": "<sampling-site token as
  published upstream>"}}`.
- `timestamp`: the upstream sampling date normalized to ISO `YYYY-MM-DD`. The source column
  is hand-typed free text; values that cannot be read without guessing (about 1.9% of the
  images) are null. `Latitude` / `Longitude` are resolved from the site token for about 87%
  of the images and null for the rest — never a guessed coordinate.

## License

Released under **{frepj_layout.LICENSE_NAME}** (`{frepj_layout.LICENSE}`) —
<{frepj_layout.LICENSE_URL}>.

## Citation

{frepj_layout.CITATION_APA}

- Paper DOI: <https://doi.org/{frepj_layout.PAPER_DOI}>
- Data DOI: <https://doi.org/{frepj_layout.DATA_DOI}>

### BibTeX

```bibtex
{frepj_layout.CITATION_BIBTEX}```
"""
    return header + body


def build_card(existing_metadata: dict | None = None):
    """Return the FREPJ ``DatasetCard`` built offline from :func:`_card_content`."""
    from huggingface_hub import DatasetCard

    return DatasetCard(_card_content(existing_metadata))


def _existing_card_metadata(repo_id: str, token: str) -> dict:
    """The YAML metadata of the card currently on the Hub, or ``{}`` when there is none."""
    from huggingface_hub import DatasetCard
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        return dict(DatasetCard.load(repo_id, repo_type="dataset", token=token).data.to_dict())
    except (EntryNotFoundError, RepositoryNotFoundError):
        return {}


def push_card(repo_id: str = TARGET_REPO_ID, token: str | None = None) -> None:
    """Preflight, then push the FREPJ dataset card (README.md) to ``repo_id``.

    The card already on the Hub is read first so its ``configs`` / ``dataset_info``
    metadata (written by ``push_to_hub``) is carried over rather than dropped.
    """
    preflight(repo_id)
    token = _resolve_token(token)
    existing = _existing_card_metadata(repo_id, token)
    card = build_card(existing)
    kept = sorted(set(existing) - set(_OWNED_CARD_METADATA))
    logger.info(f"Pushing the FREPJ dataset card to «{repo_id}» (carrying over existing metadata keys {kept}).")
    card.push_to_hub(repo_id, repo_type="dataset", token=token)


def make_public(repo_id: str = TARGET_REPO_ID, token: str | None = None, confirm_public: bool = False) -> None:
    """Flip ``repo_id`` PUBLIC — GATED behind ``confirm_public``; never auto-run.

    Raises ``ValueError`` (before any preflight, token read, or network call) unless
    ``confirm_public`` is explicitly ``True``. This is the milestone's outward-facing-action
    gate: a backgrounded agent can never make the dataset public.
    """
    if not confirm_public:
        raise ValueError(
            "Refusing to flip «{}» PUBLIC without confirm_public=True. The private->public flip is a gated, "
            "developer-confirmed action (pass --confirm-public on the CLI).".format(repo_id)
        )
    preflight(repo_id)
    token = _resolve_token(token)
    from huggingface_hub import HfApi

    logger.info(f"Flipping «{repo_id}» PUBLIC (developer-confirmed).")
    HfApi().update_repo_settings(repo_id=repo_id, repo_type="dataset", private=False, token=token)
    logger.info(f"«{repo_id}» is now PUBLIC.")


def tag_release(repo_id: str = TARGET_REPO_ID, tag: str = DEFAULT_TAG, token: str | None = None) -> None:
    """Preflight, then create Hub tag ``tag`` on the current revision of ``repo_id``.

    Refuses (``exist_ok=False``) to move a tag that already exists: re-tagging a different
    commit silently would make an existing version name point at different data.
    """
    preflight(repo_id)
    token = _resolve_token(token)
    from huggingface_hub import HfApi

    logger.info(f"Tagging the current revision of «{repo_id}» as «{tag}».")
    HfApi().create_tag(repo_id, tag=tag, tag_message=f"FREPJ-only {INTERMEDIATE_NOTE}: {tag}", repo_type="dataset", token=token)


def _exit_now_after_smoke() -> None:
    """Flush and hard-exit: a completed smoke-load otherwise hangs the interpreter at shutdown.

    Verified 2026-08-25 (logs/frepj_smoke_diag.log): the streamed first example arrives in
    ~30 s and the check passes, but normal interpreter exit then blocks indefinitely — no
    Python thread is left alive (only daemon tqdm monitors), so it is the native runtime
    behind the streaming download being torn down. Nothing is pending once the check has
    logged its verdict, so skipping Python's shutdown loses nothing. Only the smoke path
    does this; every other step exits normally.
    """
    import logging

    logger.info("Smoke-load finished; exiting immediately (the streaming runtime hangs at interpreter shutdown).")
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def publish_frepj_only(
    dataset_path: str = DEFAULT_DATASET_PATH,
    repo_id: str = TARGET_REPO_ID,
    private: bool = True,
    confirm_public: bool = False,
    hf_token: str | None = None,
) -> None:
    """Full flow: preflight -> load-from-disk -> push (PRIVATE by default) -> push card -> gated public flip.

    The PUBLIC flip happens ONLY when ``confirm_public=True``; the default (``private=True``,
    ``confirm_public=False``) pushes privately and NEVER goes public.
    """
    preflight(repo_id)
    token = _resolve_token(hf_token)
    ds = load_built(dataset_path)
    logger.info(f"Publishing the FREPJ-only dataset from «{dataset_path}» to «{repo_id}» (private={private}).")
    _push_dataset(ds, repo_id, token, private=private)
    push_card(repo_id, token=token)
    if confirm_public:
        make_public(repo_id, token=token, confirm_public=True)
    else:
        logger.info("confirm_public=False -> the repo keeps its pushed visibility; NO public flip.")


def _build_parser() -> argparse.ArgumentParser:
    """Build the staged publish CLI (each phase is a separate flag so the orchestrator runs it visibly)."""
    parser = argparse.ArgumentParser(description="Frozen-repo-guarded FREPJ-only publish helper (ORCHESTRATOR-RUN).")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="local built FREPJ-only dataset (save_to_disk)")
    parser.add_argument("--repo-id", default=TARGET_REPO_ID, help="publish target (allowlisted to planktonzilla-frepj)")
    parser.add_argument("--push-private", action="store_true", help="load-from-disk -> preflight -> push PRIVATE")
    parser.add_argument("--smoke", action="store_true", help="stream one example and assert the FREPJ columns")
    parser.add_argument("--push-card", action="store_true", help="push the FREPJ dataset card (README.md)")
    parser.add_argument("--card-only", action="store_true", help="alias for --push-card (push only the card)")
    parser.add_argument("--publish", action="store_true", help="full flow: push (PRIVATE) -> card (no public flip)")
    parser.add_argument("--make-public", action="store_true", help="flip the dataset PUBLIC (requires --confirm-public)")
    parser.add_argument(
        "--public", action="store_true", help="explicit public intent for --publish (requires --confirm-public)"
    )
    parser.add_argument("--confirm-public", action="store_true", help="explicit confirmation gate for any public flip")
    parser.add_argument(
        "--tag", nargs="?", const=DEFAULT_TAG, default=None, help=f"tag the pushed revision (default {DEFAULT_TAG})"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch the requested publish phases.

    The public flip happens ONLY when an explicit public-intent flag (``--public`` for
    ``--publish``, or ``--make-public`` on its own) is combined with ``--confirm-public``.
    ``--confirm-public`` alone — e.g. ``--publish --confirm-public`` without ``--public`` —
    NEVER flips public (WR-02).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Any public-exposing request must carry the explicit gate.
    if args.public and not args.confirm_public:
        parser.error("--public requires --confirm-public (the private->public flip is gated).")
    if args.publish and args.confirm_public and not args.public:
        parser.error("--confirm-public with --publish also requires --public (explicit public intent).")

    did = False
    smoked = False
    if args.push_private:
        push_private(load_built(args.dataset_path), args.repo_id)
        did = True
    if args.smoke:
        smoke_load(args.repo_id)
        did = smoked = True
    if args.push_card or args.card_only:
        push_card(args.repo_id)
        did = True
    if args.publish:
        # confirm_public is gated on args.public too: --publish --confirm-public alone
        # (no explicit --public intent) must never flip the repo public.
        publish_frepj_only(
            args.dataset_path,
            args.repo_id,
            private=not args.public,
            confirm_public=args.public and args.confirm_public,
        )
        did = True
    if args.tag:
        tag_release(args.repo_id, args.tag)
        did = True
    if args.make_public:
        make_public(args.repo_id, confirm_public=args.confirm_public)
        did = True
    if not did:
        parser.print_help()

    if smoked:
        _exit_now_after_smoke()


if __name__ == "__main__":
    main()
