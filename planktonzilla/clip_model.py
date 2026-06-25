"""
(c) Inria

CLIP-backed image classifier compatible with the Hugging Face `Trainer`.

Defines `ClipClassifier`, which adapts an `open_clip` visual encoder into an HF
image classifier by attaching a linear classification head, dispatching between a
ViT visual tower and a timm trunk. All `open_clip` access goes through the
`planktonzilla.open_clip_ext` seam.
"""

import torch.nn as nn
from transformers.modeling_outputs import ImageClassifierOutput

from planktonzilla import open_clip_ext


class ClipClassifier(nn.Module):
    """An `open_clip` visual encoder wrapped with a linear classification head.

    Adapts a CLIP visual tower into an HF-compatible image classifier. The
    backbone is built through the `open_clip_ext` seam and dispatched on its
    visual-tower kind:

    - **ViT path**: the visual tower's projection (`proj`) is dropped and the
        tower is composed with a fresh `nn.Linear` head as an `nn.Sequential`.
    - **timm path**: the timm trunk is extracted from the open_clip wrapper and
        its `head` is replaced with a fresh `nn.Linear`; the trunk is stored
        directly as the model.

    `forward` returns an `ImageClassifierOutput` so the module plugs into the HF
    `Trainer` (with the project's `compute_loss_func` losses computing the loss).
    """

    def __init__(
        self,
        name: str,
        pretrained: str,
        *,
        repo_path: str | None = None,
        num_features: int,
        num_labels: int,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        """Build the CLIP backbone and attach a `num_labels`-way linear head.

        Args:
            name: open_clip model name (e.g. `"ViT-B-16"`).
            pretrained: open_clip pretrained tag (e.g. `"openai"`); an `"openai"`
                tag forces QuickGELU to match the original CLIP weights (see the
                compat-shim comment below).
            repo_path: Keyword-only. When given, the model is created from this
                local/remote repo path instead of the `(name, pretrained)` pair.
            num_features: Feature dimension feeding the linear head.
            num_labels: Number of output classes (linear head width).
            id2label: Optional label-id → name mapping, stored for downstream use.
            label2id: Optional name → label-id mapping, stored for downstream use.
        """
        super().__init__()

        # Compat shim: open-clip-torch >= 3.x dropped the implicit
        # "openai pretrained tag -> quick_gelu=True" default. The OpenAI CLIP
        # weights were trained with QuickGELU; loading them into the modern
        # GELU-default model produces NaN gradients (~0.44 max-abs activation
        # delta on ViT-B-16, fully empirically verified via SMOKE-01).
        # See planktonzilla/.planning/phases/03-cutover-smoke/03-02-FAIL-SUMMARY.md.
        extra_kwargs = {}
        if pretrained and "openai" in pretrained:
            extra_kwargs["force_quick_gelu"] = True

        if repo_path:
            clip_model, _, _ = open_clip_ext.create_model_and_transforms(repo_path, **extra_kwargs)

        else:
            clip_model, _, _ = open_clip_ext.create_model_and_transforms(name, pretrained, **extra_kwargs)

        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = num_labels

        self.name_or_path = name + pretrained

        # Work on a local before self.model assignment — visual_tower_kind() needs
        # a narrowed nn.Module type (nn.Module.__setattr__ widens to Tensor | Module).
        visual = clip_model.visual
        kind = open_clip_ext.visual_tower_kind(visual)
        if kind == "vit":
            visual.proj = None  # Delete the projection
            self.model = nn.Sequential(visual, nn.Linear(num_features, num_labels))
        else:
            # kind == "timm" — visual_tower_kind() raises TypeError on unknown,
            # so we don't need a fallback or default branch.
            visual = visual.trunk
            visual.head = nn.Linear(num_features, num_labels)
            self.model = visual

    def forward(self, pixel_values, labels=None, output_attentions=None, output_hidden_states=None, return_dict=True):
        """Run the backbone and head, returning classification logits.

        Dispatches on the backbone built in `__init__`: the ViT path runs the
        visual tower then the linear head, while the timm path runs the trunk
        (which internally chains features → pooling → head). `labels`,
        `output_attentions`, and `output_hidden_states` are accepted for HF
        `Trainer` call-signature compatibility but are not used here (loss is
        computed externally; hidden states/attentions are not surfaced).

        Args:
            pixel_values: Batch of input images, shape `(B, C, H, W)`.
            labels: Unused; accepted for HF compatibility.
            output_attentions: Unused; accepted for HF compatibility.
            output_hidden_states: Unused; accepted for HF compatibility.
            return_dict: When `False`, return the tuple `(None, logits, None, None)`
                instead of an `ImageClassifierOutput`.

        Returns:
            `ImageClassifierOutput` with `logits` of shape `(B, num_labels)` (and
            `loss`/`hidden_states`/`attentions` set to `None`), or the equivalent
            tuple when `return_dict` is `False`.
        """
        if isinstance(self.model, nn.Sequential):
            features = self.model[0](pixel_values)
            logits = self.model[1](features)
        else:
            # timm-trunk path: __init__ stripped the open_clip wrapper and stored
            # the timm trunk directly (with our nn.Linear head). The trunk's
            # __call__ chains forward_features -> pooling -> head, returning
            # logits of shape (B, num_labels). Surfaced by WIRE-02 once eva02
            # was un-skipped from SMOKE-05; the prior `self.model.trunk(...)`
            # referenced a wrapper that no longer exists.
            logits = self.model(pixel_values)

        if not return_dict:
            return (None, logits, None, None)

        return ImageClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,  # (features,),
            attentions=None,
        )
