import io
import os
import tarfile

from datasets import DatasetDict, load_from_disk
from PIL import Image
from tqdm import tqdm

# Paths relative to the repository so we don't depend on a specific cluster.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "planktonzilla_17M_only_plankton")
SHARDS_DIR = os.path.join(REPO_ROOT, "data", "shards")


def export_to_tar_shards(dataset_dict, output_dir="data", shard_size=1_000):
    """
    Export a DatasetDict to .tar shards for CLIP/WebDataset-style training,
    making sure all images are saved in RGB format.
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
                    i = i_abs - start # Relative index within the shard

                    # --- 1. Image (key: image_{i}.jpg) ---
                    img = example["image"]
                    if not isinstance(img, Image.Image):
                        raise ValueError(f"The 'image' field at index {i_abs} is not a PIL.Image object")

                    # Convert to RGB before saving to cover grayscale (L),
                    # palette (P) or RGBA, since JPEG only supports RGB.
                    img_rgb = img.convert('RGB')

                    # Save the image as JPEG into a bytes buffer.
                    img_bytes = io.BytesIO()
                    img_rgb.save(img_bytes, format="JPEG", quality=95)
                    img_bytes.seek(0)

                    # Create the TarInfo and add the image file
                    img_info = tarfile.TarInfo(name=f"image_{i}.jpg")
                    img_info.size = len(img_bytes.getbuffer())
                    tar.addfile(img_info, img_bytes)

                    # --- 2. Label/Text (key: text_{i}.txt) ---
                    label_str = str(taxo_classes[example["label"]])
                    label_bytes = io.BytesIO(label_str.encode("utf-8"))

                    # Create the TarInfo and add the text file
                    label_info = tarfile.TarInfo(name=f"image_{i}.txt")
                    label_info.size = len(label_bytes.getbuffer())
                    tar.addfile(label_info, label_bytes)


def main():
    dataset = load_from_disk(INPUT_DIR)

    # Export train and validation. The validation split in the DatasetDict is
    # called "validation"; we map it to the "val" folder so it matches the
    # --val-data data/shards/val of the CLIP training flow.
    # (accepts the "validation"/"val" alias depending on how the dataset was saved).
    val_key = "validation" if "validation" in dataset else "val"
    export_to_tar_shards(
        DatasetDict({"train": dataset["train"], "val": dataset[val_key]}),
        output_dir=SHARDS_DIR,
    )

    print("DONE")


if __name__ == "__main__":
    main()
