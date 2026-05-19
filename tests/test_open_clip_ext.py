"""
(c) Inria

tests/test_open_clip_ext.py — Phase 3 SMOKE-03 + SMOKE-04 fixtures for the
open_clip externalization milestone.

SMOKE-03 (test_state_dict_keys_match_vendored): builds two FRESH ClipClassifier
instances — one using the vendored open_clip (via sys.path manipulation) and one
using the override layer (PyPI open-clip-torch via planktonzilla.open_clip_ext).
Asserts their state_dict().keys() are identical except for the new classifier head.
Defends PITFALLS P2 — silent random-init via state_dict prefix drift after
strict=False load.

The original SMOKE-03 design (compare against the pre-refactor HF Hub checkpoint
at project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt) was based on the
audit's incorrect assumption that the checkpoint was an HF-format ClipClassifier.
Reality: the checkpoint is an open_clip-native CLIP-ViT-B-16 (visual + text
towers), structurally incompatible with the ClipClassifier wrapper. The
vendored-vs-override pattern below is the correct apples-to-apples comparison
and catches real drift between vendored and upstream attention/visual implementations.

SMOKE-04 (test_preprocessing_pixel_equivalence): asserts that the override-layer
preprocessing pipeline (via planktonzilla.open_clip_ext.create_model_and_transforms)
produces tensors within torch.allclose(atol=1e-6) of the vendored open_clip
preprocessing on a canonical input image. Defends PITFALLS P7 — torchvision
antialias-default flip and other silent transform drift. The audit
(docs/open_clip_audit.md Q3) confirmed no current divergence between vendored
and upstream v3.3.0; this fixture defends the principle going forward.

Both tests are decorated @skip_in_github_ci because they require loading
open_clip models (~330 MB cached HF download) — too heavy for CI per the
project convention in tests/test_train.py.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pickle
import sys
import warnings
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from planktonzilla.clip_model import ClipClassifier
from planktonzilla.open_clip_ext import create_model_and_transforms
from planktonzilla.open_clip_ext.factory import load_checkpoint

from .shared import skip_in_github_ci

# Phase 5 (DEL-01) deleted the vendored open_clip/ directory. Tests that
# compare vendored-vs-override (SMOKE-03 + SMOKE-04) become degenerate
# post-deletion — sys.path.insert to the missing path is a no-op, and
# `import open_clip` falls back to the installed PyPI package on both
# sides of the comparison. Skip these tests cleanly when the vendored
# tree is gone; their defense-in-depth value applied only during the
# Phase 3 cutover window when both code paths coexisted.
_VENDORED_OPEN_CLIP_EXISTS = (root / "open_clip" / "src").exists()
skip_if_vendored_deleted = pytest.mark.skipif(
    not _VENDORED_OPEN_CLIP_EXISTS,
    reason="vendored open_clip/ was deleted in Phase 5 (DEL-01); "
    "vendored-vs-override comparison is no longer meaningful. Defense "
    "fired during the Phase 3 cutover window (state_dict + pixel checks "
    "passed). Either delete this test in a future cleanup or replace it "
    "with a single-pipeline structural invariant check.",
)


def _build_classifier_with_vendored_open_clip(name, pretrained, num_features, num_labels):
    """Build a ClipClassifier using the VENDORED open_clip (sys.path-injected).

    Used by SMOKE-03 and SMOKE-04 to materialize the pre-cutover code path inside
    a single pytest invocation. Mutates sys.path + sys.modules; the caller is
    responsible for the try/finally restore (see usage below).

    Returns (model, returned_transform) — the transform is the train-side
    preprocess from open_clip.create_model_and_transforms.
    """
    vendored_path = str(root / "open_clip" / "src")
    sys.path.insert(0, vendored_path)
    import open_clip as vendored_open_clip  # noqa: F811 — deliberate reimport
    # Apply the SAME QuickGELU compat shim that planktonzilla/clip_model.py uses
    # post-cutover, so the comparison is fair (both ends use the same activation).
    extra_kwargs = {"force_quick_gelu": True} if pretrained and "openai" in pretrained else {}
    clip_model, _, vendored_preprocess = vendored_open_clip.create_model_and_transforms(
        name, pretrained, **extra_kwargs
    )

    # Replicate the cutover-equivalent visual-head wiring without going through
    # ClipClassifier (which imports the OVERRIDE layer's _introspection). We need
    # this to live entirely inside the vendored import scope.
    import torch.nn as nn

    visual = clip_model.visual
    # Detect ViT vs timm using the vendored module's own classes:
    if hasattr(visual, "proj"):
        visual.proj = None
        wrapped = nn.Sequential(visual, nn.Linear(num_features, num_labels))
    else:
        visual = visual.trunk
        visual.head = nn.Linear(num_features, num_labels)
        wrapped = visual

    return wrapped, vendored_preprocess


def _restore_sys_state(original_modules):
    """Pop all open_clip-named modules and restore the previously-cached ones.

    Mandatory cleanup helper — without it, subsequent tests in the same pytest
    session would see vendored open_clip cached in sys.modules and silently
    use it instead of the installed PyPI package.
    """
    vendored_path = str(root / "open_clip" / "src")
    if vendored_path in sys.path:
        sys.path.remove(vendored_path)
    for k in [k for k in sys.modules if k.startswith("open_clip")]:
        del sys.modules[k]
    for k, v in original_modules.items():
        sys.modules[k] = v


@skip_in_github_ci
@skip_if_vendored_deleted
def test_state_dict_keys_match_vendored():
    """SMOKE-03: post-cutover ClipClassifier state_dict keys match a vendored-built equivalent.

    Builds two ClipClassifier-wrapped visual towers — one via the override layer
    (PyPI open-clip-torch through planktonzilla.open_clip_ext) and one via the
    vendored open_clip — using identical constructor arguments. The state_dict
    backbone keys must be identical. The new classifier head (1.weight/1.bias
    for ViT path) is the only legitimate divergence.

    Defends PITFALLS P2 — state_dict key prefix drift causing silent random-init
    under strict=False load.
    """
    # OVERRIDE-LAYER path: fresh ClipClassifier through PyPI open-clip-torch.
    override_model = ClipClassifier(
        name="ViT-B-16",
        pretrained="openai",
        repo_path=None,
        num_features=768,
        num_labels=2,
    )
    # ClipClassifier wraps the nn.Sequential as `self.model`, so all weights live
    # under a `model.` prefix (e.g. `model.0.class_embedding`). The vendored helper
    # returns the inner nn.Sequential directly (`0.class_embedding`). Strip the
    # outer wrapper prefix to compare apples to apples; the real drift signal is
    # in the inner key structure, not the wrapper level.
    override_keys = {k.removeprefix("model.") for k in override_model.state_dict().keys()}

    # VENDORED path: build the equivalent using sys.path-injected vendored open_clip.
    # Stash any currently-imported open_clip modules so we can restore them.
    original_modules = {k: v for k, v in sys.modules.items() if k.startswith("open_clip")}
    for k in list(original_modules):
        del sys.modules[k]
    try:
        vendored_model, _ = _build_classifier_with_vendored_open_clip(
            name="ViT-B-16",
            pretrained="openai",
            num_features=768,
            num_labels=2,
        )
        vendored_keys = set(vendored_model.state_dict().keys())
    finally:
        _restore_sys_state(original_modules)

    # The classifier head (the wrapper's added Linear at index "1") is what
    # legitimately may differ across builds — exclude both candidate head-key
    # patterns from the comparison.
    HEAD_KEY_PATTERNS = ("classifier", "1.weight", "1.bias", "head.weight", "head.bias")

    def is_head_key(k):
        return any(p in k for p in HEAD_KEY_PATTERNS)

    override_non_head = {k for k in override_keys if not is_head_key(k)}
    vendored_non_head = {k for k in vendored_keys if not is_head_key(k)}

    missing_in_override = vendored_non_head - override_non_head
    unexpected_in_override = override_non_head - vendored_non_head

    assert not missing_in_override, (
        f"Vendored backbone keys missing in override-layer state_dict: "
        f"{sorted(missing_in_override)}. This indicates state_dict prefix drift — "
        f"loading a pre-cutover checkpoint with strict=False would silently "
        f"random-initialize these layers (PITFALLS P2)."
    )
    assert not unexpected_in_override, (
        f"Unexpected backbone keys in override-layer state_dict not present "
        f"vendored-side: {sorted(unexpected_in_override)}. The cutover added "
        f"module wrapping that wasn't there before (PITFALLS P2)."
    )


@skip_in_github_ci
@skip_if_vendored_deleted
def test_preprocessing_pixel_equivalence():
    """SMOKE-04: override-layer preprocessing matches vendored open_clip preprocessing pixelwise.

    Defends PITFALLS P7 — torchvision antialias-default flip and other silent
    transform drift. The audit (docs/open_clip_audit.md Q3) confirmed no current
    divergence between vendored and upstream v3.3.0; this fixture defends the
    principle going forward.

    TODO Phase 5: when open_clip/ is deleted, replace this with a single-pipeline
    shape/dtype check or delete this test.
    """
    # Generate a deterministic canonical input. Use int64 for the arange to avoid
    # uint8 overflow on indices > 255, then % 256 + cast to uint8.
    pixels = (torch.arange(224 * 224 * 3, dtype=torch.int64) % 256).to(torch.uint8).reshape(224, 224, 3)
    pil_image = Image.fromarray(pixels.numpy())

    # OVERRIDE-LAYER path (PyPI open_clip via planktonzilla.open_clip_ext):
    _, _, override_preprocess = create_model_and_transforms("ViT-B-16", "openai")
    override_tensor = override_preprocess(pil_image)

    # VENDORED path: temporarily put vendored open_clip first on sys.path so
    # `import open_clip` resolves to the vendored copy, build a fresh transform
    # there, then restore sys.path + sys.modules. The try/finally is mandatory
    # (PITFALLS Pitfall 4) — a mid-test crash would otherwise leave the test
    # process with vendored open_clip cached in sys.modules and pollute
    # subsequent tests in the same pytest session.
    original_modules = {k: v for k, v in sys.modules.items() if k.startswith("open_clip")}
    for k in list(original_modules):
        del sys.modules[k]
    vendored_path = str(root / "open_clip" / "src")
    sys.path.insert(0, vendored_path)
    try:
        import open_clip as vendored_open_clip  # noqa: F811 — deliberate reimport
        _, _, vendored_preprocess = vendored_open_clip.create_model_and_transforms("ViT-B-16", "openai")
        vendored_tensor = vendored_preprocess(pil_image)
    finally:
        _restore_sys_state(original_modules)

    # Both tensors should be float32 of shape (3, 224, 224) and pixelwise-equal.
    assert override_tensor.shape == vendored_tensor.shape, (
        f"shape mismatch: override {override_tensor.shape}, vendored {vendored_tensor.shape}"
    )
    assert torch.allclose(override_tensor, vendored_tensor, atol=1e-6), (
        f"pixel divergence between vendored and override-layer preprocessing — "
        f"max abs diff: {(override_tensor - vendored_tensor).abs().max().item()}"
    )


def test_load_checkpoint_weights_only_retry():
    """SMOKE-02 retry safeguard: load_checkpoint retries with weights_only=False
    on pickle.UnpicklingError and emits a DeprecationWarning.

    Defends PITFALLS P4 — open_clip#998/#966: legacy .bin checkpoints containing
    numpy scalars or other non-tensor pickled objects fail under torch.load's
    weights_only=True default (torch>=2.4). The retry preserves backward
    compatibility with older checkpoints; the warning surfaces the deprecation
    so the user knows to re-save as safetensors.

    Mock-based unit test (no network, no real model weights). Verifies:
      1. First call uses weights_only=True (the safe default).
      2. On pickle.UnpicklingError, retry uses weights_only=False.
      3. A DeprecationWarning is emitted on the retry path.
      4. The successful result of the retry is returned.
    """
    call_log = []

    def fake_open_clip_load_checkpoint(model, checkpoint_path, *, strict, weights_only, device):
        call_log.append({"weights_only": weights_only})
        if weights_only is True:
            raise pickle.UnpicklingError("simulated legacy .bin numpy-scalar failure")
        return {"missing_keys": [], "unexpected_keys": []}

    with patch("planktonzilla.open_clip_ext.factory.open_clip.load_checkpoint", side_effect=fake_open_clip_load_checkpoint):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            result = load_checkpoint(model=None, checkpoint_path="/fake/legacy.bin")

    # 1+2: two calls, first weights_only=True, second weights_only=False.
    assert len(call_log) == 2, f"expected 2 attempts, got {len(call_log)}: {call_log}"
    assert call_log[0]["weights_only"] is True, "first call should use safe default (weights_only=True)"
    assert call_log[1]["weights_only"] is False, "retry should use weights_only=False"

    # 3: DeprecationWarning fired.
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1, f"expected DeprecationWarning, got: {[(w.category, str(w.message)) for w in caught]}"
    msg = str(dep_warnings[0].message)
    assert "weights_only=True" in msg and "weights_only=False" in msg, (
        f"warning message should explain both states: {msg!r}"
    )
    assert "/fake/legacy.bin" in msg, f"warning should include the checkpoint path: {msg!r}"

    # 4: result propagated from the successful retry.
    assert result == {"missing_keys": [], "unexpected_keys": []}


def test_load_checkpoint_does_not_retry_when_caller_passed_weights_only_false():
    """SMOKE-02 retry safeguard guardrail: explicit weights_only=False from caller
    skips the retry layer entirely. Failures propagate as-is.

    Prevents an infinite-retry scenario if a genuinely corrupt checkpoint hits
    UnpicklingError even at weights_only=False (e.g. truncated file).
    """
    call_log = []

    def always_fails(model, checkpoint_path, *, strict, weights_only, device):
        call_log.append({"weights_only": weights_only})
        raise pickle.UnpicklingError("corrupt file simulation")

    with patch("planktonzilla.open_clip_ext.factory.open_clip.load_checkpoint", side_effect=always_fails):
        try:
            load_checkpoint(model=None, checkpoint_path="/fake/corrupt.bin", weights_only=False)
            raise AssertionError("expected UnpicklingError to propagate")
        except pickle.UnpicklingError:
            pass

    # Single attempt — no retry since caller already opted out of the safe default.
    assert len(call_log) == 1, f"expected 1 attempt (no retry), got {len(call_log)}: {call_log}"
    assert call_log[0]["weights_only"] is False
