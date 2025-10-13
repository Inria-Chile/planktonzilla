"""
(c) Inria
"""

import os
import re
import shutil
import stat
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copy2, copytree, move, rmtree
from typing import ClassVar, Dict, Final, Optional, Union
from zipfile import ZipFile

import aiohttp
import numpy as np
import plotext as plt
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_dataset_builder,
)
from datasets.download import DownloadConfig, DownloadManager
from huggingface_hub import DatasetCard
from humanize import naturalsize
from PIL import Image
from rich import print as rich_print
from rich.markdown import Markdown
from tqdm import tqdm

import planktonzilla.dataset_import.public_data as public_data
from planktonzilla.dataset import compute_mean_and_std_dev
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

DATACARD_TEMPLATE = """
---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---
# Dataset Card for *{{ pretty_name | default("Dataset Name", true) }}*
{{ dataset_description | default("[More Information Needed]", true) }}

- **Original dataset available online at:**  <{{ source_url | default("[More Information Needed]", true)}}>.
- **Original dataset license:** <{{ license | default("[More Information Needed]", true)}}>.

## Details

- **`train` split means (RGB):** {{ dataset_means | default("[More Information Needed]", true) }}
- **`train` split standard deviations (RGB):** {{ dataset_stds | default("[More Information Needed]", true) }}

{{ report_markdown | default("[More Information Needed]", true) }}

## Reference
{{ citation_apa | default("[More Information Needed]", true)}}

### BibTEX
```bibtex
{{ citation_bibtex | default("[More Information Needed]", true)}}
```

## Usage
```python
from datasets import load_dataset

dataset = load_dataset("{{hf_org_name}}/{{hf_dataset_name}}")
```
"""


def is_dir_empty(dir: Path) -> bool:
    if dir and dir.exists() and os.listdir(dir):
        return False
    return True


def strip_ansi_codes(text):
    """
    Removes ANSI escape sequences from a string.
    """
    reaesc = re.compile(r"\x1b[^m]*m")
    return reaesc.sub("", text)


def report_dataset_content(huggingface_dataset: Dataset | DatasetDict) -> str:
    def report_split(dataset: Dataset, split_name: str) -> str:
        class_idxs, class_counts = np.unique(dataset["label"], return_counts=True)

        content = []
        for class_idx in class_idxs:
            class_name = dataset.features["label"].int2str(int(class_idx))
            content += [f"{class_idx}: {class_name}"]

        plt.simple_bar(content, class_counts.astype(int), title=f"Label histogram for {split_name} split ", width=83)
        plt.show()

        return strip_ansi_codes(plt.build())

    if isinstance(huggingface_dataset, DatasetDict):
        split_reports = []
        split_reports = [
            f"**Samples per class for split `{split}`**\n ```{report_split(huggingface_dataset[split], split)}```\n"
            for split in huggingface_dataset
        ]
        return "\n".join(split_reports)
    else:
        return report_split(huggingface_dataset) + "\n"


def unzip(zip_file: Path, output_dir: Path, show_progress: bool = True):
    """Unzips a zip file showing progress.

    Args:
        zip_file (Path): file to unzip
        output_dir (Path): where to put results
    """
    with ZipFile(zip_file, "r") as zip_ref:
        for file in tqdm(
            iterable=zip_ref.namelist(),
            total=len(zip_ref.namelist()),
            desc=f"Extracting {zip_file.name} ({naturalsize(os.stat(zip_file).st_size)})",
            leave=False,
            disable=not show_progress,
        ):
            zip_ref.extract(member=file, path=output_dir)


def cleanup_imagefolder_empty_dirs(imagefolder_dir: Path) -> None:
    """Delete empty subfolders as torchvision ImageFolder crashes if a folder is empty."""
    for class_dir in os.listdir(imagefolder_dir):
        dir = imagefolder_dir / class_dir
        if dir.is_dir() and not os.listdir(dir):
            shutil.rmtree(dir)


def is_valid_image_file(image_filename):
    try:
        with Image.open(image_filename) as img:
            # img.verify() seems not to be enough to check all cases,
            # cropping image should do.
            img.crop((5, 5, 5, 5))
        return True
    except (IOError, SyntaxError):
        return False


@dataclass
class DatasetImporter:
    data_dir: Path

    human_readable_name: str = None
    download_uris: list[str] = None

    push_to_hub: Optional[bool] = False
    hf_dataset_name: Optional[str] = None
    hf_private: Optional[bool] = True
    hf_token: str = None
    hf_org_name: str = None

    show_progress: Optional[bool] = True
    num_proc: Optional[int] = cpu_count()

    # download-related configs
    force_download: Optional[bool] = False
    resume_download: Optional[bool] = True
    force_imagefolder_preparation: Optional[bool] = True
    max_download_retries: Optional[int] = 5
    http_timeout: Optional[int] = 3600
    push_to_hub_retries: Optional[int] = 10
    check_image_file_integrity: Optional[bool] = False

    # if we have manually downloaded the files add the archives here
    manual_download_local_file_names: str | list[str] = None

    cleanup_after_processing: Optional[bool] = False

    description: str = ""
    license: str = None
    citation_bibtex: str = None
    citation_apa: str = None
    source_url: str = None
    image_url: str = None
    paperswithcode_id: str = None
    arxiv_id: str = None

    def _validate(self):
        if self.push_to_hub:
            if not self.hf_token:
                raise ValueError("push_to_hub=True but hf_token is not set.")
            if not self.hf_dataset_name:
                raise ValueError("push_to_hub=True but hgfc_dataset_name is not set.")

    def __post_init__(self):
        self._validate()
        self.data_dir = Path(self.data_dir)

        self.imagefolder_dir = self.data_dir / f"{self.__class__.__name__.lower()}_imagefolder"
        self.raw_dir = self.data_dir / f"{self.__class__.__name__.lower()}_raw_download"
        self.extracted_dirs = None
        self.download_manager = None
        self.hf_dataset = None

    def _download_and_extract(self):
        self.download_manager = DownloadManager(
            base_path=self.raw_dir,
            data_dir=self.raw_dir,
            download_config=DownloadConfig(
                cache_dir=self.raw_dir,
                force_download=self.force_download,
                resume_download=self.resume_download,
                max_retries=self.max_download_retries,
                num_proc=self.num_proc,
                disable_tqdm=not self.show_progress,
                storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=self.http_timeout)}},
            ),
        )
        if self.manual_download_local_file_names:
            logger.info(f"Using manually downloaded file {self.manual_download_local_file_names}.")
            downloaded_paths = self.manual_download_local_file_names
        else:
            logger.info(f"Downloading files to {self.raw_dir}.")
            downloaded_paths = self.download_manager.download(self.download_uris)

        logger.info("Extracting file(s).")
        self.extracted_dirs = self.download_manager.extract(downloaded_paths)

    def _prepare_imagefolder(self):
        raise NotImplementedError()

    def update_dataset_metadata(self):
        logger.info(f"Updating «{self.hf_org_name}/{self.hf_dataset_name}» card metadata.")
        card = DatasetCard.load(self.hf_org_name + "/" + self.hf_dataset_name)

        card.data.dataset_info["description"] = self.description
        card.data.dataset_info["dataset_name"] = self.human_readable_name
        card.data.dataset_info["citation"] = self.citation_bibtex
        card.data.dataset_info["homepage"] = self.source_url

        card.data["pretty_name"] = self.human_readable_name
        card.data["dataset_description"] = self.description
        card.data["license"] = self.license
        card.data["source_url"] = self.source_url

        if self.paperswithcode_id:
            card.data["paperswithcode_id"] = self.paperswithcode_id

        if self.arxiv_id:
            card.data["arxiv_id"] = self.arxiv_id

        card.data["citation_bibtex"] = self.citation_bibtex
        card.data["citation_apa"] = self.citation_apa
        card.data["task_categories"] = ["image-classification"]
        card.data["hf_dataset_name"] = self.hf_dataset_name
        card.data["hf_org_name"] = self.hf_org_name

        if not self.hf_dataset:
            self.hf_dataset = load_dataset(self.hf_org_name + "/" + self.hf_dataset_name)

        card.data["report_markdown"] = report_dataset_content(self.hf_dataset)

        means, stds = compute_mean_and_std_dev(self.hf_dataset["train"])
        card.data["dataset_means"] = "[" + ", ".join([str(item) for item in means]) + "]"
        card.data["dataset_stds"] = "[" + ", ".join([str(item) for item in stds]) + "]"

        new_card = DatasetCard.from_template(card.data, template_str=DATACARD_TEMPLATE)
        new_card.push_to_hub(self.hf_org_name + "/" + self.hf_dataset_name)

    def show_details(self):
        builder = load_dataset_builder(self.hf_org_name + "/" + self.hf_dataset_name)
        rich_print(builder.info)

        card = DatasetCard.load(self.hf_org_name + "/" + self.hf_dataset_name)
        rich_print(Markdown(card.text))

    def _push_to_hub(self):
        if self.push_to_hub:
            if self.hf_dataset:
                logger.info(
                    f"Pushing «{self.human_readable_name}» to HuggingFace Hub as «{self.hf_org_name}/{self.hf_dataset_name}»."
                )
                for attempt in range(self.push_to_hub_retries):
                    try:
                        self.hf_dataset.push_to_hub(
                            self.hf_org_name + "/" + self.hf_dataset_name,
                            token=self.hf_token,
                            private=self.hf_private,
                        )
                        break
                    except Exception as e:
                        logger.warning(
                            f"Push to hub attempt {attempt + 1}/{self.push_to_hub_retries} failed, retrying. Cause: {e}."
                        )
                self.update_dataset_metadata()
            else:
                logger.error("No dataset to push.")
        else:
            logger.warning("Skipping pushing dataset to HuggingFace Hub, set push_to_hub=True to change this.")

    def cleanup(self):
        if self.cleanup_after_processing:
            logger.info("Removing downloaded and intermediate files.")
            if self.download_manager:
                self.download_manager.delete_extracted_files()

            if self.raw_dir and self.raw_dir.exists():
                rmtree(self.raw_dir, ignore_errors=True)

            if self.imagefolder_dir and self.imagefolder_dir.exists():
                rmtree(self.imagefolder_dir, ignore_errors=True)
        else:
            logger.info("Keeping downloaded and intermediate files, set cleanup_after_processing=True to change this.")

    def import_dataset(self) -> Union[Dataset, DatasetDict]:
        self._download_and_extract()

        if not self.force_imagefolder_preparation and not is_dir_empty(self.imagefolder_dir):
            logger.info(
                f"ImageFolder dir {self.imagefolder_dir} is not empty, using its content as dataset. Set force_imagefolder_preparation=True to change this."  # noqa: E501
            )
        else:
            logger.info(f"Preparing dataset as imagefolder in {self.imagefolder_dir}")
            self._prepare_imagefolder()

        if self.check_image_file_integrity:
            for class_dir in tqdm(
                os.listdir(self.imagefolder_dir),
                desc="Validating classes.",
                disable=not self.show_progress,
                leave=False,
            ):
                for file in tqdm(
                    os.listdir(self.imagefolder_dir / class_dir),
                    disable=not self.show_progress,
                    leave=False,
                ):
                    if not is_valid_image_file(self.imagefolder_dir / class_dir / file):
                        logger.warning("Invalid file {file} in class {class_dir} detected. Removing it from dataset.")
                        os.remove(self.imagefolder_dir / class_dir / file)

        logger.info(f"Loading imagefolder in {self.imagefolder_dir} as HuggingFace dataset.")
        self.hf_dataset = load_dataset(
            "imagefolder",
            data_dir=self.imagefolder_dir,
            name=self.hf_dataset_name,
            save_infos=True,
            token=self.hf_token,
            num_proc=self.num_proc,
        )
        self._push_to_hub()
        self.cleanup()


class LenslessDatasetImporter(DatasetImporter):
    DATASET_FILENAME: Final[str] = "lensless_dataset"

    def _download_and_extract(self):
        dataset_path = Path(public_data.__path__[0])

        logger.info(f"Unzipping lensless zip {dataset_path / (self.DATASET_FILENAME + '.zip')}.")

        unzip(
            dataset_path / (self.DATASET_FILENAME + ".zip"),
            self.raw_dir,
            show_progress=self.show_progress,
        )
        self.extracted_dirs = self.raw_dir / self.DATASET_FILENAME

    def _prepare_imagefolder(self):
        if self.imagefolder_dir.exists():
            rmtree(self.imagefolder_dir, ignore_errors=True)
        self.imagefolder_dir.mkdir(exist_ok=True, parents=True)
        copytree(self.extracted_dirs, self.imagefolder_dir, dirs_exist_ok=True)
        (self.imagefolder_dir / "TRAIN_IMAGE").rename(self.imagefolder_dir / "train")
        (self.imagefolder_dir / "TEST_IMAGE").rename(self.imagefolder_dir / "test")


class ZooLakeDatasetImporter(DatasetImporter):
    SPLIT_NAMES: ClassVar[Dict[str, str]] = {
        "train_split": "train_filenames.txt",
        "val_split": "val_filenames.txt",
        "test_split": "test_filenames.txt",
    }

    def _prepare_imagefolder(self):
        for split_name in tqdm(
            list(self.SPLIT_NAMES),
            desc="Processing original split",
            leave=False,
            position=0,
            disable=not self.show_progress,
        ):
            with open(
                Path(self.extracted_dirs) / "data" / "zoolake_train_test_val_separated" / self.SPLIT_NAMES[split_name]
            ) as f:
                lines = f.readlines()
                for line in tqdm(
                    lines,
                    desc=f"Moving files in {split_name}",
                    leave=False,
                    position=1,
                    disable=not self.show_progress,
                ):
                    _, _, _, class_name, folder, file_name = line.strip().split("/")

                    source_img_file = Path(self.extracted_dirs) / "data" / "zooplankton_0p5x" / class_name / folder / file_name

                    target_folder = self.imagefolder_dir / split_name / class_name

                    if not (source_img_file).exists():
                        logger.warning(f"In split {split_name} (class {class_name}) file {source_img_file} does not exist.")
                        continue

                    target_folder.mkdir(exist_ok=True, parents=True)

                    if (target_folder / file_name).exists():
                        logger.warning(f"File name duplicate {file_name}, skipping.")
                    else:
                        copy2(
                            source_img_file,
                            target_folder,
                        )


class ZooScanNetDatasetImporter(DatasetImporter):
    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "ZooScanNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class WHOIPlanktonDatasetImporter(DatasetImporter):
    def _prepare_imagefolder(self):
        for release_folder in tqdm(
            self.extracted_dirs,
            desc="ImageFolder move progress",
            leave=False,
            position=0,
            disable=not self.show_progress,
        ):
            for folder in tqdm(
                [item for item in (self.raw_dir / release_folder).glob("*") if item.is_dir()],
                desc=f"Moving release {release_folder}",
                leave=False,
                position=1,
                disable=not self.show_progress,
            ):
                (self.imagefolder_dir / folder.name).mkdir(exist_ok=True)
                for img_file in folder.glob("*.png"):
                    try:
                        copy2(folder / img_file, self.imagefolder_dir / folder.name)
                    except OSError:
                        logger.debug(f"File {folder / img_file} already in {self.imagefolder_dir / folder.name}.")
            rmtree(self.raw_dir / release_folder, ignore_errors=True)


class JEDISystemsOceansCPICSDatasetImporter(DatasetImporter):
    def _prepare_imagefolder(self) -> None:
        for zip_file in tqdm(
            sorted((Path(self.extracted_dirs) / "CPICS_Validated").glob("*.zip")),
            desc="Unzip progress",
            leave=False,
            disable=not self.show_progress,
        ):
            unzip(
                zip_file,
                Path(self.extracted_dirs) / "CPICS_Validated",
                show_progress=self.show_progress,
            )

            # nested zip files are an intermedite results, we delete them to save space
            Path(zip_file).unlink()

        # fixing file permissions issue in nested zips
        for file in (Path(self.extracted_dirs) / "CPICS_Validated").glob("*"):
            file.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IWUSR)  # owner read/write/excecute

        for release_dir in tqdm(
            sorted([item for item in (Path(self.extracted_dirs) / "CPICS_Validated").glob("*") if item.is_dir()]),
            desc="ImageFolder preparation",
            leave=False,
            position=0,
            disable=not self.show_progress,
        ):
            for class_folder in tqdm(
                [item for item in release_dir.glob("*") if item.is_dir()],
                desc=f"Moving release {release_dir.name}",
                leave=False,
                position=1,
                disable=not self.show_progress,
            ):
                (self.imagefolder_dir / class_folder.name).mkdir(exist_ok=True, parents=True)
                for img_file in class_folder.glob("*.png"):
                    try:
                        move(
                            class_folder / img_file,
                            self.imagefolder_dir / class_folder.name,
                        )
                    except OSError:
                        logger.debug(f"File {class_folder / img_file} already in {self.imagefolder_dir / class_folder.name}.")
            rmtree(release_dir, ignore_errors=True)


class UVP6NetDatasetImporter(DatasetImporter):
    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class ZooCAMNetDatasetImporter(DatasetImporter):
    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "ZooCamNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class FlowCAMNetDatasetImporter(DatasetImporter):
    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "FlowCamNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class ISIISNetDatasetImporter(DatasetImporter):
    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "ISIISNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)
