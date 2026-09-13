"""
(c) Inria

The WebDataset export had no test at all, which is how both defects below survived: neither
raises, neither logs, and the `.tar` files they produce open cleanly.

  * **Sample keys were unique only within a shard.** WebDataset takes a member's basename as the
    sample's ``__key__``, and the writer numbered from 0 in every shard — so a 20-shard split
    published twenty samples answering to ``image_0``. Pairing the ``.jpg`` with its ``.txt``
    only needs the two basenames to match inside one tar, which is why nothing ever failed; what
    breaks is anything that keys BY the key.
  * **An empty split exported nothing and said nothing.** The split folder is still created, so
    the run looks like it worked, and the absence surfaces later as a training job with no data.

Offline and tiny: four 2x2 images through a real ``datasets.Dataset``, sharded two per tar, read
back out of the real ``tarfile``. No fixture on disk, no network.
"""

import io
import tarfile
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features
from datasets import Image as HFImage
from PIL import Image

from planktonzilla.planktonzilla_dataset.save_planktonzilla_for_clip import export_to_tar_shards

CLASSES = ["copepoda", "detritus"]


def _dataset(n: int):
    """``n`` distinguishable 2x2 images, alternating between the two classes."""
    return Dataset.from_dict(
        {
            "image": [Image.new("RGB", (2, 2), (index * 40 % 256, 0, 0)) for index in range(n)],
            "label": [index % len(CLASSES) for index in range(n)],
        },
        features=Features({"image": HFImage(), "label": ClassLabel(names=CLASSES)}),
    )


def _keys(split_dir: Path):
    """``{shard name: [member names]}`` for every tar the export wrote."""
    return {path.name: sorted(tarfile.open(path).getnames()) for path in sorted(split_dir.glob("*.tar"))}


@pytest.fixture
def exported(tmp_path):
    export_to_tar_shards(DatasetDict({"train": _dataset(4)}), output_dir=str(tmp_path), shard_size=2)
    return tmp_path / "train"


def test_a_sample_key_is_unique_across_the_whole_split(exported):
    """Two shards, four samples, four distinct keys — not two keys used twice.

    Before the fix the two shards were byte-for-byte confusable: both held ``image_0.*`` and
    ``image_1.*``. A consumer joining model output back to samples by ``__key__``, or
    deduplicating a resampled stream, silently collapsed the split to its first shard's worth.
    """
    shards = _keys(exported)
    assert len(shards) == 2, shards

    stems = [name.rsplit(".", 1)[0] for names in shards.values() for name in names]
    assert len(set(stems)) == 4, f"keys repeat across shards: {sorted(stems)}"
    assert sorted(set(stems)) == ["image_000000000", "image_000000001", "image_000000002", "image_000000003"]


def test_the_pair_a_sample_is_made_of_still_shares_one_basename(exported):
    """The property the per-shard counter was there to satisfy, which must survive the fix.

    WebDataset groups members into a sample by basename; a ``.jpg`` and ``.txt`` that disagree
    yield two half-samples rather than one whole one.
    """
    for shard, names in _keys(exported).items():
        stems = [name.rsplit(".", 1)[0] for name in names]
        assert len(names) == 4, f"{shard}: expected two pairs, got {names}"
        assert sorted(set(stems)) == sorted(stems[::2]), f"{shard}: unpaired members in {names}"
        for stem in set(stems):
            assert f"{stem}.jpg" in names and f"{stem}.txt" in names


def test_the_key_order_follows_the_dataset_order(exported):
    """Zero-padded, so a lexical sort of the keys is the dataset's own order.

    ``image_10`` sorting before ``image_2`` is the ordinary consequence of an unpadded counter,
    and it silently reorders any consumer that sorts by key.
    """
    stems = sorted(name.rsplit(".", 1)[0] for names in _keys(exported).values() for name in names)
    assert stems == sorted(stems, key=lambda stem: int(stem.split("_")[1]))


def test_a_label_is_written_as_its_class_name_not_its_index(exported):
    """The ``.txt`` side is the caption CLIP trains against; an integer would train on digits."""
    first = min(exported.glob("*.tar"))
    with tarfile.open(first) as tar:
        captions = {tar.extractfile(name).read().decode("utf-8") for name in tar.getnames() if name.endswith(".txt")}
    assert captions <= set(CLASSES) and captions


def test_an_image_is_re_encoded_as_rgb_jpeg(exported):
    """Stated in the docstring as the reason for the re-encode; nothing checked it."""
    first = min(exported.glob("*.tar"))
    with tarfile.open(first) as tar:
        name = next(name for name in tar.getnames() if name.endswith(".jpg"))
        image = Image.open(io.BytesIO(tar.extractfile(name).read()))
    assert image.format == "JPEG"
    assert image.mode == "RGB"


def test_an_empty_split_is_reported_rather_than_silently_skipped(tmp_path, caplog):
    """`n_shards` is 0, the loop body never runs, and the split folder still appears.

    So the export "succeeded" and produced a directory a training job will happily point at and
    find nothing in. An empty split is a real outcome of an upstream filter matching nothing, so
    it warns rather than raising.
    """
    with caplog.at_level("WARNING"):
        export_to_tar_shards(DatasetDict({"train": _dataset(0)}), output_dir=str(tmp_path), shard_size=2)

    assert list((tmp_path / "train").glob("*.tar")) == []
    assert any("is empty" in record.getMessage() for record in caplog.records), caplog.text


def test_a_split_shorter_than_one_shard_still_exports(tmp_path):
    """The off-by-one neighbours of the empty case: 1 sample and shard_size 2 is one short shard."""
    export_to_tar_shards(DatasetDict({"train": _dataset(1)}), output_dir=str(tmp_path), shard_size=2)

    shards = _keys(tmp_path / "train")
    assert list(shards) == ["shard_00000.tar"]
    assert shards["shard_00000.tar"] == ["image_000000000.jpg", "image_000000000.txt"]
