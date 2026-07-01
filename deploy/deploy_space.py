"""
(c) Inria

Deploy the planktonzilla explorer as a Hugging Face Gradio Space.

This is the Phase-14 (SPACE-03) deploy helper. It uploads a CURATED, secret-free
subset of the repo — only what ``planktonzilla/app.py`` imports at runtime — to a
Gradio Space, then (behind an explicit ``--confirm-public`` gate) flips it public.

Design / safety:
- **Explicit allowlist** (``CURATED``): we upload exactly the files the app needs,
  never the whole repo. No ``.env``, no tokens/keys, no ``.planning/``, no
  ``uv.lock``, no ``.git``, no training stack. A secret-exclusion gate re-scans the
  staged tree and aborts if anything suspicious slipped in.
- **Package paths preserved**: both CSVs resolve via ``Path(__file__).parent`` in
  their modules, so the curated tree keeps the ``planktonzilla/...`` layout.
- **Private-first**: create private, cold-start smoke, then ``--make-public
  --confirm-public``. The public flip NEVER happens without both flags.
- The public HF dataset needs no token at runtime; the deploy uses the caller's
  cached HF login (write access to the target org required).

Stepwise usage (run from the repo root):
    uv run --group explorer python deploy/deploy_space.py --stage        # build + audit staging dir
    uv run --group explorer python deploy/deploy_space.py --smoke-local  # build_demo() offline
    uv run --group explorer python deploy/deploy_space.py --create       # create PRIVATE space
    uv run --group explorer python deploy/deploy_space.py --upload       # upload curated tree
    uv run --group explorer python deploy/deploy_space.py --status       # runtime stage
    uv run --group explorer python deploy/deploy_space.py --make-public --confirm-public
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ID = "project-oceania/planktonzilla-explorer"
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- The curated allowlist: repo-relative paths uploaded to the Space -----------------
# Only what planktonzilla/app.py imports at runtime (verified import graph + Path(__file__) CSV resolution).
CURATED: list[str] = [
    "planktonzilla/app.py",
    "requirements.txt",
    "planktonzilla/__init__.py",
    "planktonzilla/explorer/__init__.py",
    "planktonzilla/explorer/shapes.py",
    "planktonzilla/explorer/data_access.py",
    "planktonzilla/explorer/sankey.py",
    "planktonzilla/explorer/hierarchy.py",
    "planktonzilla/explorer/geomap.py",
    "planktonzilla/explorer/data/inferred_dataset_locations.csv",
    "planktonzilla/planktonzilla_dataset/__init__.py",
    "planktonzilla/planktonzilla_dataset/constants.py",
    "planktonzilla/planktonzilla_dataset/planktonzilla_taxonomy.csv",
    "planktonzilla/utils/__init__.py",
    "planktonzilla/utils/logger.py",
]
# The Space README (with the HF YAML header) is uploaded AS README.md at the Space root.
SPACE_README_SRC = "deploy/README.md"

# --- Secret-exclusion gate ------------------------------------------------------------
# Filename patterns that must NEVER appear in the staged tree.
FORBIDDEN_NAME_RE = re.compile(
    r"(^|/)\.env|secret|token|\.key$|\.pem$|credential|id_rsa|\.git(/|$)|\.planning(/|$)|uv\.lock$",
    re.IGNORECASE,
)
# Content patterns that look like leaked credentials (HF tokens, private keys, AWS keys).
FORBIDDEN_CONTENT_RE = re.compile(
    r"hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}",
)


def stage() -> Path:
    """Copy the curated allowlist into a temp staging dir (preserving paths); audit it."""
    staging = Path(tempfile.mkdtemp(prefix="pz-space-"))
    for rel in CURATED:
        src = REPO_ROOT / rel
        if not src.is_file():
            sys.exit(f"FATAL: curated file missing: {rel}")
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # Space README -> README.md at root
    readme_src = REPO_ROOT / SPACE_README_SRC
    if not readme_src.is_file():
        sys.exit(f"FATAL: Space README missing: {SPACE_README_SRC}")
    shutil.copy2(readme_src, staging / "README.md")

    _audit(staging)
    print(f"STAGED {len(CURATED) + 1} files at {staging}")
    return staging


def _audit(staging: Path) -> None:
    """Fail closed if any staged file trips the name or content secret gate."""
    offenders: list[str] = []
    for p in sorted(staging.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(staging).as_posix()
        if FORBIDDEN_NAME_RE.search(rel):
            offenders.append(f"name:{rel}")
            continue
        # Scan textual files for credential-shaped content (CSV/py/txt/md only).
        if p.suffix.lower() in {".py", ".txt", ".md", ".csv", ".toml", ".cfg"}:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if FORBIDDEN_CONTENT_RE.search(text):
                offenders.append(f"content:{rel}")
    if offenders:
        sys.exit("SECRET-EXCLUSION GATE TRIPPED — refusing to deploy:\n  " + "\n  ".join(offenders))
    print("secret-exclusion gate: PASS (no secrets in staged tree)")


def smoke_local() -> None:
    """Build the Gradio Blocks offline (imports resolve, 4 tabs) — no launch, no network."""
    sys.path.insert(0, str(REPO_ROOT))
    import planktonzilla.app as app

    demo = app.build_demo()
    # Blocks built without launching or hitting the network (geomap read is deferred).
    print(f"local smoke: build_demo() OK -> {type(demo).__name__}")


def create() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    print(f"created (private) space: {REPO_ID}")


def upload() -> None:
    from huggingface_hub import HfApi

    staging = stage()
    api = HfApi()
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=str(staging),
        commit_message="Deploy planktonzilla explorer (curated subset)",
    )
    print(f"uploaded curated tree to {REPO_ID}")


def status() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.space_info(REPO_ID)
    runtime = getattr(info, "runtime", None)
    stage_ = getattr(runtime, "stage", None) if runtime else None
    print(f"space {REPO_ID}: private={info.private} runtime.stage={stage_}")


def make_public(confirm: bool) -> None:
    if not confirm:
        sys.exit("REFUSING to make public without --confirm-public (private->public is gated).")
    from huggingface_hub import HfApi

    api = HfApi()
    api.update_repo_settings(repo_id=REPO_ID, repo_type="space", private=False)
    print(f"space {REPO_ID} is now PUBLIC")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", action="store_true", help="build + audit the curated staging dir")
    ap.add_argument("--smoke-local", action="store_true", help="build_demo() offline")
    ap.add_argument("--create", action="store_true", help="create the PRIVATE space")
    ap.add_argument("--upload", action="store_true", help="upload the curated tree")
    ap.add_argument("--status", action="store_true", help="print the space runtime stage")
    ap.add_argument("--make-public", action="store_true", help="flip the space public (needs --confirm-public)")
    ap.add_argument("--confirm-public", action="store_true", help="explicit confirmation for the public flip")
    args = ap.parse_args()

    did = False
    if args.stage:
        stage()
        did = True
    if args.smoke_local:
        smoke_local()
        did = True
    if args.create:
        create()
        did = True
    if args.upload:
        upload()
        did = True
    if args.status:
        status()
        did = True
    if args.make_public:
        make_public(args.confirm_public)
        did = True
    if not did:
        ap.print_help()


if __name__ == "__main__":
    main()
