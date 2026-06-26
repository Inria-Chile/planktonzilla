"""
(c) Inria

Download, normalize, and push public plankton source datasets to HuggingFace Hub.

Defines the :class:`DatasetImporter` abstract base class and one concrete
subclass per source dataset (ZooLake, ZooScanNet, WHOI, EcoTaxa-derived sets,
etc.). The shared lifecycle is: download + extract the raw archive, normalize it
into a torchvision-style imagefolder layout (the per-source ``_prepare_imagefolder``
hook), load it as a HuggingFace ``imagefolder`` dataset, and — when gated on by
``push_to_hub`` — push the dataset and a rendered dataset card to the Hub.

This module also holds the dataset-card Jinja template and small image/zip/IO
helpers used during normalization.
"""

import concurrent.futures
import csv
import gzip
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
# Dataset *{{ pretty_name | default("Dataset Name", true) }}*
{{ dataset_description | default("[More Information Needed]", true) }}

- **Original dataset available online at:**  <{{ source_url | default("[More Information Needed]", true)}}>.
- **Original dataset license:** <{{ license | default("[More Information Needed]", true)}}>.

## Details

- **train split means (RGB):** {{ dataset_means | default("[More Information Needed]", true) }}
- **train split standard deviations (RGB):** {{ dataset_stds | default("[More Information Needed]", true) }}

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
    """Return True if ``dir`` is None, missing, or contains no entries."""
    if dir and dir.exists() and os.listdir(dir):
        return False
    return True


def strip_ansi_codes(text):
    """
    Removes ANSI escape sequences from a string.
    """
    reaesc = re.compile(r"\x1b[^m]*m")
    return reaesc.sub("", text)


def copytree_filtered(src: Path, dst: Path):
    """Copy a directory tree into ``dst``, skipping macOS junk (``._*``, ``.DS_Store``).

    Merges into ``dst`` if it already exists (``dirs_exist_ok=True``).
    """
    copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("._*", ".DS_Store"),
    )


def report_dataset_content(huggingface_dataset: Dataset | DatasetDict) -> str:
    """Build a Markdown report of per-class sample counts for each split.

    Renders an ANSI-free ``plotext`` bar chart of the label histogram per split,
    suitable for embedding in the dataset card. Accepts a single ``Dataset`` or a
    ``DatasetDict`` (one section per split).

    Returns:
        The Markdown report string.
    """

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
    """Return True if PIL can open and crop the file, i.e. it is a readable image.

    A crop is used rather than ``Image.verify()`` because ``verify`` misses some
    corruption cases; cropping forces the decoder to touch pixel data.
    """
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
    """Abstract base for importing a public plankton dataset into HuggingFace format.

    Each concrete subclass corresponds to one source dataset and need only implement
    :meth:`_prepare_imagefolder`, which normalizes that source's extracted layout into
    a torchvision-style imagefolder (``<root>/<split>/<class>/*`` or
    ``<root>/<class>/*``). The base class drives the shared lifecycle in
    :meth:`import_dataset`: download + extract, prepare the imagefolder, optionally
    validate image integrity, load it as a HuggingFace ``imagefolder`` dataset, push to
    the Hub (gated on ``push_to_hub``), then clean up intermediate files.

    Subclasses may also override :meth:`_download_and_extract` when the source is not a
    plain set of ``download_uris`` (e.g. a bundled zip).

    Key fields:
        data_dir: Root directory for raw downloads and the prepared imagefolder.
        human_readable_name: Display name used in the dataset card.
        download_uris: Source URLs to download and extract.
        push_to_hub: Master gate for any Hub push; when False the push is skipped.
        hf_dataset_name / hf_org_name / hf_token / hf_private: Hub target + auth.
        force_download / resume_download / force_imagefolder_preparation: Re-run gates
            controlling whether downloads and imagefolder preparation are redone.
        manual_download_local_file_names: Pre-downloaded archives to use instead of
            fetching ``download_uris``.
        check_image_file_integrity: When True, drop unreadable images before loading.
        cleanup_after_processing: When True, remove raw/intermediate files at the end.
        description / license / citation_* / source_url / paperswithcode_id / arxiv_id:
            Dataset-card metadata.

    Instance attributes set in ``__post_init__``: ``imagefolder_dir`` and ``raw_dir``
    (both derived from ``data_dir`` and the lowercased class name), plus
    ``extracted_dirs``, ``download_manager``, and ``hf_dataset`` placeholders.
    """

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
        """Validate that Hub credentials/target are present when ``push_to_hub`` is set.

        Raises:
            ValueError: If ``push_to_hub`` is True but ``hf_token`` or
                ``hf_dataset_name`` is missing.
        """
        if self.push_to_hub:
            if not self.hf_token:
                raise ValueError("push_to_hub=True but hf_token is not set.")
            if not self.hf_dataset_name:
                raise ValueError("push_to_hub=True but hgfc_dataset_name is not set.")

    def __post_init__(self):
        """Validate config and derive the per-class raw/imagefolder paths.

        ``imagefolder_dir`` and ``raw_dir`` are namespaced by the lowercased subclass
        name so that distinct importers sharing a ``data_dir`` do not collide.
        """
        self._validate()
        self.data_dir = Path(self.data_dir)

        self.imagefolder_dir = self.data_dir / f"{self.__class__.__name__.lower()}_imagefolder"
        self.raw_dir = self.data_dir / f"{self.__class__.__name__.lower()}_raw_download"
        self.extracted_dirs = None
        self.download_manager = None
        self.hf_dataset = None

    def _download_and_extract(self):
        """Download ``download_uris`` (or use manual files) and extract them.

        Builds an aiohttp-backed ``DownloadManager`` honoring the download config
        fields (force/resume, retries, timeout, num_proc). When
        ``manual_download_local_file_names`` is set, those archives are used in place
        of fetching ``download_uris``. Sets ``self.download_manager`` and
        ``self.extracted_dirs`` as a side effect.
        """
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
        """Normalize this source's extracted layout into an imagefolder (subclass hook).

        Concrete importers must override this to move/copy ``self.extracted_dirs`` into
        ``self.imagefolder_dir`` as ``<split>/<class>/*`` (or ``<class>/*`` when the
        source has no splits).

        Raises:
            NotImplementedError: Always, on the abstract base.
        """
        raise NotImplementedError()

    def update_dataset_metadata(self):
        """Render and push the dataset card for the already-published Hub dataset.

        Loads the existing card, refreshes metadata (description, citations, license,
        per-split label histogram, and train-split RGB mean/std), renders it through
        ``DATACARD_TEMPLATE``, and pushes the new card to the Hub. Loads the dataset
        from the Hub if not already cached on ``self.hf_dataset``. Side effects:
        network reads (card + dataset) and a Hub card push.
        """
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
        """Pretty-print the Hub dataset's builder info and rendered card to the terminal.

        Reads the dataset builder info and dataset card from the Hub (network reads).
        """
        builder = load_dataset_builder(self.hf_org_name + "/" + self.hf_dataset_name)
        rich_print(builder.info)

        card = DatasetCard.load(self.hf_org_name + "/" + self.hf_dataset_name)
        rich_print(Markdown(card.text))

    def _push_to_hub(self):
        """Push the prepared dataset to the Hub, gated on ``push_to_hub``.

        No-op (logs a warning) when ``push_to_hub`` is False. Otherwise pushes
        ``self.hf_dataset`` with up to ``push_to_hub_retries`` attempts, then refreshes
        the dataset card via :meth:`update_dataset_metadata`. Logs an error if there is
        no dataset to push. Side effects: Hub dataset + card push.
        """
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
        """Remove raw downloads and extracted files, gated on ``cleanup_after_processing``.

        No-op (logs an info message) when ``cleanup_after_processing`` is False. The
        prepared imagefolder is intentionally kept (the removal is commented out) so it
        can be reused on a later run. Side effects: deletes ``raw_dir`` and extracted
        files from disk.
        """
        if self.cleanup_after_processing:
            logger.info("Removing downloaded and intermediate files.")
            if self.download_manager:
                self.download_manager.delete_extracted_files()

            if self.raw_dir and self.raw_dir.exists():
                rmtree(self.raw_dir, ignore_errors=True)

            # if self.imagefolder_dir and self.imagefolder_dir.exists():
            #    rmtree(self.imagefolder_dir, ignore_errors=True)
        else:
            logger.info("Keeping downloaded and intermediate files, set cleanup_after_processing=True to change this.")

    def import_dataset(self) -> Union[Dataset, DatasetDict]:
        """Run the full import lifecycle for this source dataset.

        Steps: (re)build the imagefolder when missing or
        ``force_imagefolder_preparation`` is set — downloading + extracting and calling
        the subclass :meth:`_prepare_imagefolder`; optionally drop unreadable images
        when ``check_image_file_integrity``; load the imagefolder as a HuggingFace
        ``imagefolder`` dataset (mapping ``train``/``validation``/``val``/``test``
        subdirectories to splits, or a single ``train`` split when none exist); then
        push to the Hub (gated on ``push_to_hub``) and clean up.

        Side effects: network downloads, extensive disk writes under ``data_dir``, and
        an optional Hub push. Sets ``self.hf_dataset``.

        Raises:
            RuntimeError: If extraction failed and no raw data is available to prepare
                the imagefolder.
        """
        imagefolder_exists = not is_dir_empty(self.imagefolder_dir)
        raw_exists = self.raw_dir.exists() and bool(os.listdir(self.raw_dir))

        need_to_build_imagefolder = not imagefolder_exists or self.force_imagefolder_preparation

        if need_to_build_imagefolder:
            if raw_exists:
                logger.info(f"Raw data already exists at {self.raw_dir}, resolving extracted paths from cache.")
            else:
                logger.info("Downloading and extracting dataset.")

            # Si los archivos ya están en el raw_dir,
            # no los descargará de nuevo; solo leerá la caché y asignará la ruta
            self._download_and_extract()

            if getattr(self, "extracted_dirs", None) is None:
                raise RuntimeError("Cannot prepare imagefolder: extraction failed or raw data is unavailable.")

            logger.info(f"Preparing dataset as imagefolder in {self.imagefolder_dir}")
            self._prepare_imagefolder()

        else:
            logger.info(
                f"Using existing imagefolder at {self.imagefolder_dir}. Set force_imagefolder_preparation=True to rebuild."
            )

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

        root = Path(self.imagefolder_dir)

        split_aliases = {
            "train": ["train"],
            "validation": ["validation", "val"],
            "test": ["test"],
        }

        data_files = {}

        for canonical_split, aliases in split_aliases.items():
            for alias in aliases:
                split_path = root / alias
                if split_path.exists():
                    data_files[canonical_split] = str(split_path / "*/[!._]*")
                    break

        # fallback: dataset sin splits
        if not data_files:
            data_files = {"train": str(root / "*/*[!._]*")}

        self.hf_dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            name=self.hf_dataset_name,
            save_infos=True,
            token=self.hf_token,
            num_proc=self.num_proc,
        )

        self._push_to_hub()
        self.cleanup()


class LenslessDatasetImporter(DatasetImporter):
    """Importer for the bundled lensless plankton dataset.

    Unlike URL-based sources, this dataset ships as a zip inside the
    ``planktonzilla.dataset_import.public_data`` package, so it overrides
    :meth:`_download_and_extract` to unzip locally. Its ``TRAIN_IMAGE`` / ``TEST_IMAGE``
    folders are renamed to the canonical ``train`` / ``test`` splits.
    """

    DATASET_FILENAME: Final[str] = "lensless_dataset"

    def _download_and_extract(self):
        """Unzip the bundled lensless dataset from ``public_data`` into ``raw_dir``."""
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
    """Importer for the ZooLake dataset (lake zooplankton, predefined splits).

    The source ships ``train``/``val``/``test`` filename manifests that index into a
    flat ``zooplankton_0p5x`` image tree; :meth:`_prepare_imagefolder` reads those
    manifests (see :attr:`SPLIT_NAMES`) and copies each listed image into the matching
    ``<split>/<class>`` folder, preserving the original split assignment.
    """

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
    """Importer for ZooScanNet (single, unsplit set of per-class folders).

    Copies each class folder under ``ZooScanNet/imgs`` into the imagefolder root.
    """

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "ZooScanNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class WHOIPlanktonDatasetImporter(DatasetImporter):
    """Importer for the WHOI plankton dataset (multiple per-release archives).

    Iterates the extracted release folders and copies each release's per-class ``.png``
    images into the imagefolder root (merging classes across releases), then deletes
    each consumed release folder from ``raw_dir`` to save space.
    """

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
    """Importer for the JEDI Systems / Oceans CPICS validated dataset (nested zips).

    The ``CPICS_Validated`` payload contains nested per-release zips: this unzips each,
    deletes the now-redundant nested archive, fixes restrictive file permissions left by
    the inner zips, then moves each release's per-class ``.png`` images into the
    imagefolder root and removes the consumed release directory.
    """

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
    """Importer for UVP6Net; copies each class folder under ``imgs`` into the root."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class ZooCAMNetDatasetImporter(DatasetImporter):
    """Importer for ZooCamNet; copies each class folder under ``ZooCamNet/imgs``."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "ZooCamNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class FlowCAMNetDatasetImporter(DatasetImporter):
    """Importer for FlowCamNet; copies each class folder under ``FlowCamNet/imgs``."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "FlowCamNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class ISIISNetDatasetImporter(DatasetImporter):
    """Importer for ISIISNet; copies each class folder under ``ISIISNet/imgs``."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "ISIISNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class PlanktoScopeDatasetImporter(DatasetImporter):
    """Importer for the PlanktoScope reference set.

    Copies each class folder under ``Planktoscope_reference/imgs``, skipping non-dirs
    and macOS junk entries (``._*``, ``.DS_Store``).
    """

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "Planktoscope_reference" / "imgs").iterdir(),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            if (
                not plankton_class_dir.is_dir()
                or plankton_class_dir.name.startswith("._")
                or plankton_class_dir.name == ".DS_Store"
            ):
                continue

            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class GlobalUVP5NetDatasetImporter(DatasetImporter):
    """Importer for the Global UVP5 dataset, where labels come from a separate metadata file.

    The image archive has no class folders; instead an ``objects.tsv.gz`` (downloaded
    separately from :attr:`OBJECTS_URL`) maps each ``object_id`` to a ``taxon``.
    :meth:`_prepare_imagefolder` parses that mapping straight from the zip, creates one
    folder per taxon, then copies each image (named by object id) into its taxon folder
    using a bounded thread pool (network download + heavy disk I/O).
    """

    OBJECTS_URL = "https://www.seanoe.org/data/00964/107583/data/120871.zip"

    def _prepare_imagefolder(self):
        aux_dir = self.data_dir / "global_uvp5_aux"
        aux_dir.mkdir(parents=True, exist_ok=True)

        # --- Metadata (obj_id to taxo) ---
        dm = DownloadManager(
            base_path=aux_dir,
            data_dir=aux_dir,
            download_config=DownloadConfig(
                cache_dir=aux_dir,
                force_download=self.force_download,
                resume_download=self.resume_download,
                max_retries=self.max_download_retries,
                num_proc=self.num_proc,
                disable_tqdm=not self.show_progress,
                storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=self.http_timeout)}},
            ),
        )

        logger.info("Downloading objects metadata.")
        zip_path = dm.download(self.OBJECTS_URL)

        # --- Mapping ---
        mapping = {}
        logger.info("Parsing metadata directly from ZIP...")
        with ZipFile(zip_path, "r") as z:
            tsv_filename = next((name for name in z.namelist() if name.endswith("objects.tsv.gz")), None)
            if not tsv_filename:
                raise RuntimeError("objects.tsv.gz not found in zip")

            with z.open(tsv_filename) as gz_fileobj:
                with gzip.open(gz_fileobj, "rt", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    header = next(reader)

                    try:
                        obj_idx = header.index("object_id")
                        taxon_idx = header.index("taxon")
                    except ValueError:
                        raise RuntimeError("Columns 'object_id' or 'taxon' missing in TSV.")

                    for row in reader:
                        mapping[row[obj_idx]] = row[taxon_idx]

        logger.info("Creating target directories...")
        unique_taxa = set(mapping.values())
        for taxon in unique_taxa:
            (self.imagefolder_dir / taxon).mkdir(parents=True, exist_ok=True)

        images_root = Path(self.extracted_dirs) / "images"
        copy_tasks = []

        logger.info("Mapping files to their target directories...")

        try:
            sample_dirs = [entry.path for entry in os.scandir(images_root) if entry.is_dir()]
        except FileNotFoundError:
            raise RuntimeError(f"Directory not found: {images_root}. Check your extracted_dirs path.")

        for sample_dir_path in tqdm(sample_dirs, desc="Scanning directories", leave=False, disable=not self.show_progress):
            for entry in os.scandir(sample_dir_path):
                if not entry.is_file():
                    continue

                object_id = entry.name.rsplit(".", 1)[0]
                taxon = mapping.get(object_id)

                dst = self.imagefolder_dir / taxon / entry.name

                copy_tasks.append((entry.path, dst))

        # --- MultiThread ---
        def copy_worker(task):
            src, dst = task
            if not dst.exists():
                try:
                    copy2(src, dst)
                except OSError as e:
                    logger.warning(f"Failed to copy {src}: {e}")

        if copy_tasks:
            max_threads = min(16, self.num_proc)
            logger.info(f"Starting multi-threaded copy with {max_threads} workers...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                list(
                    tqdm(
                        executor.map(copy_worker, copy_tasks),
                        total=len(copy_tasks),
                        desc="Copying images",
                        disable=not self.show_progress,
                        leave=False,
                    )
                )
        else:
            logger.info("No new images to copy.")


class PlanktonSet1DatasetImporter(DatasetImporter):
    """Importer for PlanktonSet-1.

    Copies each class folder from the deeply nested
    ``0127422/2.3/data/0-data/FINAL_Plankton_Segments_12082014`` path, skipping non-dirs
    and dotfile/macOS junk entries.
    """

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "0127422" / "2.3" / "data" / "0-data" / "FINAL_Plankton_Segments_12082014").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            if (
                not plankton_class_dir.is_dir()
                or plankton_class_dir.name.startswith(".")
                or plankton_class_dir.name.startswith("._")
                or plankton_class_dir.name == ".DS_Store"
            ):
                continue

            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class SYKEIFCB2022DatasetImporter(DatasetImporter):
    """Importer for the SYKE IFCB 2022 set; copies class folders under ``labeled_20201020``."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "labeled_20201020").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class SYKEZooScan2024DatasetImporter(DatasetImporter):
    """Importer for the SYKE ZooScan 2024 set.

    Copies each class folder from
    ``0127422/2.3/data/FINAL_Plankton_Segments_12082014``.
    """

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "0127422" / "2.3" / "data" / "FINAL_Plankton_Segments_12082014").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)
