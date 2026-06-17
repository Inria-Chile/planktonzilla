import argparse
import io
import logging
import os
import tarfile

from datasets import DatasetDict, load_from_disk
from PIL import Image
from tqdm import tqdm

from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# Paths relative to the repository so we don't depend on a specific cluster.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "planktonzilla_17M_only_plankton")
SHARDS_DIR = os.path.join(REPO_ROOT, "data", "shards")


def export_to_tar_shards(
    dataset_dict: DatasetDict,
    output_dir: str = "data",
    shard_size: int = 1_000,
    jpeg_quality: int = 95,
) -> None:
    """Export a DatasetDict to .tar shards for CLIP/WebDataset-style training.

    All images are re-encoded as RGB JPEG so JPEG-only consumers can read them.

    Args:
        dataset_dict: Splits to export; one subdirectory of shards per split.
        output_dir: Directory where the per-split shard folders are written.
        shard_size: Maximum number of samples per .tar shard.
        jpeg_quality: JPEG quality used when re-encoding the images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, dataset in dataset_dict.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        total_samples = len(dataset)
        n_shards = (total_samples + shard_size - 1) // shard_size

        taxo_classes = dataset.features["label"].names

        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = min((shard_idx + 1) * shard_size, total_samples)

            shard_path = os.path.join(split_dir, f"shard_{shard_idx:05d}.tar")
            shard_indices = range(start, end)

            with tarfile.open(shard_path, "w") as tar:
                # Loop over the absolute indices of the dataset
                for i_abs in tqdm(shard_indices, desc=f"{split_name} shard {shard_idx}"):
                    example = dataset[i_abs]
                    i = i_abs - start  # Relative index within the shard

                    # --- 1. Image (key: image_{i}.jpg) ---
                    img = example["image"]
                    if not isinstance(img, Image.Image):
                        raise ValueError(f"The 'image' field at index {i_abs} is not a PIL.Image object")

                    # Convert to RGB before saving to cover grayscale (L),
                    # palette (P) or RGBA, since JPEG only supports RGB.
                    img_rgb = img.convert("RGB")

                    # Save the image as JPEG into a bytes buffer.
                    img_bytes = io.BytesIO()
                    img_rgb.save(img_bytes, format="JPEG", quality=jpeg_quality)
                    img_bytes.seek(0)

                    # Create the TarInfo and add the image file
                    img_info = tarfile.TarInfo(name=f"image_{i}.jpg")
                    img_info.size = len(img_bytes.getbuffer())
                    tar.addfile(img_info, img_bytes)

                    # --- 2. Label/Text (key: image_{i}.txt) ---
                    # The basename must match the image (image_{i}) so WebDataset
                    # groups the .jpg and its .txt into the same sample.
                    label_str = str(taxo_classes[example["label"]])
                    label_bytes = io.BytesIO(label_str.encode("utf-8"))

                    # Create the TarInfo and add the text file
                    label_info = tarfile.TarInfo(name=f"image_{i}.txt")
                    label_info.size = len(label_bytes.getbuffer())
                    tar.addfile(label_info, label_bytes)


def main() -> None:
    """Load the only-plankton dataset and export train/val splits to tar shards."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Saved only-plankton DatasetDict to load.")
    parser.add_argument(
        "--output-dir",
        "--shards-dir",
        dest="output_dir",
        default=SHARDS_DIR,
        help="Directory where the per-split shard folders are written.",
    )
    parser.add_argument("--shard-size", type=int, default=1_000, help="Maximum number of samples per .tar shard.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality used when re-encoding the images.")
    args = parser.parse_args()

    dataset = load_from_disk(args.input_dir)

    # Export train and validation. The validation split in the DatasetDict is
    # called "validation"; we map it to the "val" folder so it matches the
    # --val-data data/shards/val of the CLIP training flow.
    # (accepts the "validation"/"val" alias depending on how the dataset was saved).
    val_key = "validation" if "validation" in dataset else "val"
    export_to_tar_shards(
        DatasetDict({"train": dataset["train"], "val": dataset[val_key]}),
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        jpeg_quality=args.jpeg_quality,
    )

    logger.info("DONE")


if __name__ == "__main__":
    main()
