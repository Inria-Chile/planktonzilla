import torch.nn as nn
from transformers.modeling_outputs import ImageClassifierOutput

from planktonzilla import open_clip_ext


class ClipClassifier(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: str,
        repo_path: str,
        num_features: int,
        num_labels: int,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()

        if repo_path:
            clip_model, _, _ = open_clip_ext.create_model_and_transforms(repo_path)

        else:
            clip_model, _, _ = open_clip_ext.create_model_and_transforms(name, pretrained)

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
        if isinstance(self.model, nn.Sequential):
            features = self.model[0](pixel_values)
            logits = self.model[1](features)
        else:
            features = self.model.trunk(pixel_values)
            logits = self.model.head(features)

        if not return_dict:
            return (None, logits, None, None)

        return ImageClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,  # (features,),
            attentions=None,
        )
