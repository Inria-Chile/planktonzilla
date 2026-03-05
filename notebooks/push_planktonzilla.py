import sys
import os

root_path = os.path.abspath("..")
sys.path.append(root_path)

from planktonzilla.dataset_import.dataset_importer import *


from datasets import load_from_disk, concatenate_datasets, DatasetDict, Dataset, Value, load_dataset
from pathlib import Path

hf_dataset = load_from_disk("/lustre/fsn1/projects/rech/tec/uod68bo/data/planktonzilla_ood")

@dataclass
class PlanktonzillaDatasetImporter(DatasetImporter):
    def import_dataset(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self._push_to_hub()


from huggingface_hub.utils import get_token
hf_token = get_token()


# hf_dataset = DatasetDict({
#     "train": datasets,
# })


importer = PlanktonzillaDatasetImporter(
    data_dir="",
    hf_token =hf_token,
    hf_private=False,
    push_to_hub=True,
    hf_dataset_name="planktonzilla_ood",
    hf_org_name="project-oceania",
    human_readable_name="Plankton Dataset",
    description = "A dataset composed of all publicly available, labeled plankton datasets",
    source_url="https://huggingface.co/datasets/project-oceania/planktonzilla_ood",
    license="cc-by-4.0",
    citation_bibtex="""@misc{planktonzilla_ood,
        author       = {Inria Chile},
        title        = {Planktonzilla OOD dataset},
        month        = feb,
        year         = 2026,
        version      = {1.0.0},
        doi          = {TBD},
        url          = {https://huggingface.co/datasets/project-oceania/planktonzilla_ood},
    }""",
)



def main():
    importer.import_dataset(hf_dataset)
    print("DONE")
    
if __name__ == "__main__":
    main()
