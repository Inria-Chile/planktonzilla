"""
(c) Inria

tests/test_open_clip_ext.py — Phase 3 SMOKE-03 + SMOKE-04 fixtures for the
open_clip externalization milestone.

SMOKE-03 (test_state_dict_keys_match_pre_refactor): asserts that a freshly-constructed
post-cutover ClipClassifier exposes the same backbone state_dict keys (modulo the new
classifier head) as the pre-refactor HF safetensors checkpoint at
project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt. Defends PITFALLS P2 — silent
random-init via state_dict prefix drift after strict=False load.

SMOKE-04 (test_preprocessing_pixel_equivalence): asserts that the override-layer
preprocessing pipeline (via planktonzilla.open_clip_ext.create_model_and_transforms)
produces tensors within torch.allclose(atol=1e-6) of the vendored open_clip preprocessing
on a canonical input image. Defends PITFALLS P7 — torchvision antialias-default flip and
other silent transform drift. The audit (docs/open_clip_audit.md Q3) confirmed no current
divergence between vendored and upstream v3.3.0; this fixture defends the principle going
forward.

Both tests are decorated @skip_in_github_ci because they require ~330 MB of HF Hub
downloads and model construction — too heavy for CI per the project convention in
tests/test_train.py.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import sys
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file

from planktonzilla.clip_model import ClipClassifier
from planktonzilla.open_clip_ext import create_model_and_transforms

from .shared import skip_in_github_ci

PRE_REFACTOR_CHECKPOINT = "project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt"


@pytest.fixture(scope="session")
def pre_refactor_checkpoint_dir():
    """Session-scoped fixture: snapshot_download the pre-refactor HF checkpoint once.

    Per 03-RESEARCH.md Open Question 3 recommendation: shared across tests in a single
    pytest invocation to avoid re-downloading ~330 MB. Skips gracefully on network errors
    so the fixture degrades to a pytest.skip rather than a hard failure when offline.
    """
    try:
        local_dir = snapshot_download(repo_id=PRE_REFACTOR_CHECKPOINT)
    except Exception as e:  # noqa: BLE001 — fixture-time network failure → skip, not fail.
        pytest.skip(f"network required for pre-refactor checkpoint download: {e!r}")
    yield Path(local_dir)


@skip_in_github_ci
def test_state_dict_keys_match_pre_refactor(pre_refactor_checkpoint_dir):
    """SMOKE-03: post-cutover ClipClassifier state_dict keys match the pre-refactor checkpoint.

    Defends PITFALLS P2 — state_dict key prefix drift causing silent random-init under
    strict=False load. The classifier head's last-dim may legitimately differ (pre-refactor
    was trained with a different num_labels), so HEAD_KEY_PATTERNS excludes head keys from
    the comparison. The backbone keys must match exactly.
    """
    # Construct a fresh post-cutover ClipClassifier using the same args as the pre-refactor
    # checkpoint. vit-base-clip-224-openai.yaml: ViT-B-16/openai, num_features=768.
    model = ClipClassifier(
        name="ViT-B-16",
        pretrained="openai",
        repo_path=None,
        num_features=768,
        num_labels=2,
        id2label={0: "class_0", 1: "class_1"},
        label2id={"class_0": 0, "class_1": 1},
    )
    post_keys = set(model.state_dict().keys())

    # Load the pre-refactor checkpoint's state_dict keys (without instantiating the full
    # model — we only need keys, not weights).
    safetensors_path = Path(pre_refactor_checkpoint_dir) / "model.safetensors"
    pre_state = load_file(str(safetensors_path))
    pre_keys = set(pre_state.keys())

    # Classifier head keys may legitimately differ (different num_labels, different naming).
    # Exclude them from the comparison; assert the backbone-key invariant.
    HEAD_KEY_PATTERNS = ("classifier", "head", "1.weight", "1.bias")

    def is_head_key(k):
        return any(p in k for p in HEAD_KEY_PATTERNS)

    pre_non_head = {k for k in pre_keys if not is_head_key(k)}
    post_non_head = {k for k in post_keys if not is_head_key(k)}

    missing_in_post = pre_non_head - post_non_head
    unexpected_in_post = post_non_head - pre_non_head

    assert not missing_in_post, (
        f"Pre-refactor backbone keys missing in post-cutover state_dict: {sorted(missing_in_post)}. "
        f"This indicates state_dict prefix drift — loading the pre-refactor checkpoint with "
        f"strict=False would silently random-initialize these layers (PITFALLS P2)."
    )
    assert not unexpected_in_post, (
        f"Unexpected backbone keys in post-cutover state_dict not present pre-refactor: "
        f"{sorted(unexpected_in_post)}. The cutover added module wrapping that wasn't there before "
        f"(PITFALLS P2)."
    )


@skip_in_github_ci
def test_preprocessing_pixel_equivalence():
    """SMOKE-04: override-layer preprocessing matches vendored open_clip preprocessing pixelwise.

    Defends PITFALLS P7 — torchvision antialias-default flip and other silent transform drift.
    The audit (docs/open_clip_audit.md Q3) confirmed no current divergence between vendored
    and upstream v3.3.0; this fixture defends the principle going forward so any future
    upstream bump or vendored modification that DID change pixel values would surface
    immediately.

    TODO Phase 5: when open_clip/ is deleted, replace this with a single-pipeline shape/dtype
    check or delete this test.
    """
    # Generate a deterministic canonical input (no on-disk asset needed; gradient image 224x224).
    pixels = torch.arange(224 * 224 * 3, dtype=torch.uint8).reshape(224, 224, 3) % 256
    pil_image = Image.fromarray(pixels.numpy())

    # OVERRIDE-LAYER path (PyPI open_clip via planktonzilla.open_clip_ext):
    _, _, override_preprocess = create_model_and_transforms("ViT-B-16", "openai")
    override_tensor = override_preprocess(pil_image)

    # VENDORED path: temporarily put vendored open_clip first on sys.path so
    # `import open_clip` resolves to the vendored copy, build a fresh transform there,
    # then restore sys.path + sys.modules. The try/finally is mandatory (PITFALLS Pitfall 4) —
    # a mid-test crash would otherwise leave the test process with vendored open_clip cached
    # in sys.modules and pollute subsequent tests in the same pytest session.
    vendored_path = str(root / "open_clip" / "src")
    original_open_clip = sys.modules.pop("open_clip", None)
    original_modules = {k: v for k, v in sys.modules.items() if k.startswith("open_clip")}
    for k in list(original_modules):
        del sys.modules[k]
    sys.path.insert(0, vendored_path)
    try:
        import open_clip as vendored_open_clip  # noqa: F811 — deliberate reimport
        _, _, vendored_preprocess = vendored_open_clip.create_model_and_transforms("ViT-B-16", "openai")
        vendored_tensor = vendored_preprocess(pil_image)
    finally:
        if vendored_path in sys.path:
            sys.path.remove(vendored_path)
        for k in [k for k in sys.modules if k.startswith("open_clip")]:
            del sys.modules[k]
        if original_open_clip is not None:
            sys.modules["open_clip"] = original_open_clip
            for k, v in original_modules.items():
                sys.modules[k] = v

    # Both tensors should be float32 of shape (3, 224, 224) and pixelwise-equal.
    assert override_tensor.shape == vendored_tensor.shape, (
        f"shape mismatch: override {override_tensor.shape}, vendored {vendored_tensor.shape}"
    )
    assert torch.allclose(override_tensor, vendored_tensor, atol=1e-6), (
        f"pixel divergence between vendored and override-layer preprocessing — max abs diff: "
        f"{(override_tensor - vendored_tensor).abs().max().item()}"
    )
