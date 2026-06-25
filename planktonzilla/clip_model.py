"""
(c) Inria

Image-classification head on top of a CLIP visual tower.

`ClipClassifier` adapts a pretrained CLIP model (loaded through the
`planktonzilla.open_clip_ext` seam) into a plain image classifier: it keeps the
visual encoder, drops the CLIP projection, and attaches a linear classification
head sized to the dataset's number of classes. The forward pass returns a
Hugging Face `ImageClassifierOutput` so the model is a drop-in for the
`transformers` `Trainer` used elsewhere in planktonzilla.
"""

import torch.nn as nn
from transformers.modeling_outputs import ImageClassifierOutput

from planktonzilla import open_clip_ext


class ClipClassifier(nn.Module):
    """CLIP visual tower repurposed as an image classifier.

    Loads a CLIP model, strips its contrastive projection, and replaces it with a
    linear head that maps visual features to class logits. Both ViT-style towers
    (open_clip `VisionTransformer`) and timm-trunk towers are supported; the
    appropriate head wiring is selected automatically from the tower kind.
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
        """Build the classifier from a pretrained CLIP visual tower.

        Args:
            name: open_clip model name (e.g. ``"ViT-B-16"``). Ignored when
                ``repo_path`` is given.
            pretrained: open_clip pretrained tag (e.g. ``"openai"``). When the
                tag contains ``"openai"``, QuickGELU is forced to match the
                activation the original OpenAI weights were trained with.
            repo_path: Optional path/identifier to load the model from directly
                instead of the ``name``/``pretrained`` pair.
            num_features: Dimensionality of the visual features feeding the
                linear head (must match the tower's output width).
            num_labels: Number of target classes; sets the head's output size.
            id2label: Optional mapping from class index to class name, stored for
                downstream use (e.g. inference and model cards).
            label2id: Optional inverse mapping from class name to class index.
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
        """Classify a batch of images and return logits.

        Args:
            pixel_values: Batch of preprocessed image tensors, shape
                ``(B, C, H, W)``.
            labels: Accepted for `transformers` API compatibility; unused here
                (loss is computed by the `Trainer`'s loss function, not the model).
            output_attentions: Accepted for API compatibility; ignored.
            output_hidden_states: Accepted for API compatibility; ignored.
            return_dict: When ``True`` (default) return an
                `ImageClassifierOutput`; when ``False`` return a tuple
                ``(loss, logits, hidden_states, attentions)`` with everything but
                the logits set to ``None``.

        Returns:
            ImageClassifierOutput | tuple: The class logits of shape
            ``(B, num_labels)``, wrapped per ``return_dict``.
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
