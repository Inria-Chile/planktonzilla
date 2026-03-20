import torch.nn as nn
import open_clip
from transformers.modeling_outputs import ImageClassifierOutput

class ClipClassifier(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: str,
        repo_path: str,
        num_features: int,
        num_labels: int,
        id2label: dict = None,
        label2id: dict = None,
    ):
        super().__init__()

        if repo_path:
            clip_model, _, _ = open_clip.create_model_and_transforms(repo_path)

        else:
            clip_model, _, _ = open_clip.create_model_and_transforms(name, pretrained)

        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = num_labels
        
        self.name_or_path = name + pretrained
        self.model = clip_model.visual
        
        try:
            _ = self.model.proj # ViT models
            self.model.proj = None # Delete the projection

            self.model = nn.Sequential(
                self.model,
                nn.Linear(num_features, num_labels)
            )
        except:
            self.model = self.model.trunk
            self.model.head = nn.Linear(num_features, num_labels)

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
            hidden_states=None, # (features,),
            attentions=None,
        )
