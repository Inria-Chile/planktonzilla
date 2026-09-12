"""
(c) Inria

A partially built imagefolder is the failure mode this suite exists for.

WHOI and JEDI copy their sources release by release and delete each release directory as
they consume it, and ``global_uvp5`` copies 7.4 M images through a thread pool. All three
report a source "imported" from the state of the directory afterwards, so an interruption
— or a copy that failed for a reason nobody was told about — used to leave a FRACTION of a
source looking exactly like the whole of it, and the next run spliced that fraction into
the published corpus.

The property throughout: **an import that did not finish must not read as one that did.**
"""

from pathlib import Path

import pyrootutils
import pytest

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import Image

import planktonzilla.dataset_import.dataset_importer as di


def _importer(tmp_path, source="whoi-plankton", extra=()):
    """Instantiate a real source config against a temporary data_dir."""
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_partial_imports")
    cfg = hydra.compose(
        config_name="import_dataset",
        overrides=[
            f"dataset_import={source}",
            "dataset_import.push_to_hub=False",
            "dataset_import.show_progress=False",
            f"paths.data_dir={tmp_path}",
            f"dataset_import.data_dir={tmp_path}",
            *extra,
        ],
    )
    importer = hydra.utils.instantiate(cfg.dataset_import)
    GlobalHydra.instance().clear()
    return importer


def _populate(imagefolder, classes=("bosmina", "keratella")):
    for name in classes:
        (imagefolder / name).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(imagefolder / name / "a.png")


# --- the in-progress marker (finding 29) --------------------------------------------


def test_an_interrupted_preparation_does_not_read_as_complete(tmp_path):
    """Release 4 of 9: a non-empty imagefolder that is four ninths of the source.

    Before the marker, this returned True and the fraction was published as the whole.
    """
    importer = _importer(tmp_path)
    _populate(importer.imagefolder_dir)
    assert importer.imagefolder_is_complete(), "the control: this is what a finished import looks like"

    (importer.imagefolder_dir / di.IMPORT_IN_PROGRESS_MARKER).touch()

    assert not importer.imagefolder_is_complete()


def test_an_imagefolder_built_before_the_marker_existed_is_still_complete(tmp_path):
    """Backwards compatibility is the whole reason the marker signals IN PROGRESS.

    Requiring a "finished" marker instead would declare every imagefolder already on disk
    incomplete, and re-import 17 M images to learn nothing.
    """
    importer = _importer(tmp_path)
    _populate(importer.imagefolder_dir)

    assert importer.imagefolder_is_complete()
    assert not (importer.imagefolder_dir / di.IMPORT_IN_PROGRESS_MARKER).exists()


def test_the_marker_is_hidden_so_the_imagefolder_loader_ignores_it(tmp_path):
    """A visible file at the imagefolder root would reach the loader as data."""
    assert di.IMPORT_IN_PROGRESS_MARKER.startswith(".")


# --- copy failures are failures (findings 27, 13) ------------------------------------


def test_a_failed_copy_stops_the_import_instead_of_shortening_it(tmp_path):
    """The handler this replaces logged 'already in <dir>' at DEBUG — a cause that
    cannot occur, since copy2 overwrites silently. What it hid was ENOSPC/EACCES/EIO."""
    importer = _importer(tmp_path)

    with pytest.raises(RuntimeError) as failure:
        di._fail_on_copy_errors(importer, [(Path("/raw/a.png"), OSError(28, "No space left on device"))])

    message = str(failure.value)
    assert "1 image(s) could not be copied" in message
    assert "No space left on device" in message
    assert "re-run: images already copied are skipped" in message
    # WHOI and JEDI rmtree each release as they consume it, so "just re-run" is not advice they
    # can follow — the extracted tree the re-run would resume from is gone.
    assert "force_download=true" in message


def test_no_failures_is_silent(tmp_path):
    di._fail_on_copy_errors(_importer(tmp_path), [])


def test_the_error_names_the_first_failures_and_counts_the_rest(tmp_path):
    """A 7.4 M-image copy can fail thousands of times; the log must stay readable."""
    failures = [(Path(f"/raw/{i}.png"), OSError("boom")) for i in range(12)]

    with pytest.raises(RuntimeError, match=r"\(\+7 more\)"):
        di._fail_on_copy_errors(_importer(tmp_path), failures)


# --- a decompression bomb is an answer, not a crash (finding 14) ---------------------


def test_a_decompression_bomb_is_reported_invalid_rather_than_crashing(tmp_path, monkeypatch):
    """DecompressionBombError derives straight from Exception, so it escaped the
    (IOError, SyntaxError) tuple and took down the integrity walk mid-source."""
    image = tmp_path / "big.png"
    Image.new("RGB", (64, 64)).save(image)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)

    assert di.is_valid_image_file(image) is False


def test_a_normal_image_still_validates(tmp_path):
    image = tmp_path / "small.png"
    Image.new("RGB", (8, 8)).save(image)

    assert di.is_valid_image_file(image) is True


# --- a multi-file manual download is a list (finding 28) -----------------------------


def test_a_multi_file_manual_download_stays_a_list(tmp_path):
    """omegaconf's ListConfig is not a list subclass, so datasets' map_nested treated it
    as a scalar and looked up "['a.zip', 'b.zip']" as one relative path."""
    importer = _importer(tmp_path)
    importer.manual_download_local_file_names = OmegaConf.create([str(tmp_path / "a.zip"), str(tmp_path / "b.zip")])

    argument = importer.manual_download_extract_argument()

    assert type(argument) is list, "a ListConfig here is the bug"
    assert argument == [str(tmp_path / "a.zip"), str(tmp_path / "b.zip")]


def test_a_single_manual_download_stays_a_scalar(tmp_path):
    """Arity matters downstream: JEDI indexes into extracted_dirs, WHOI iterates it."""
    importer = _importer(tmp_path)
    importer.manual_download_local_file_names = str(tmp_path / "only.zip")

    assert importer.manual_download_extract_argument() == str(tmp_path / "only.zip")


def test_a_one_element_list_stays_a_list(tmp_path):
    """Declared as a list, extracted as a list — the declaration decides, not the length."""
    importer = _importer(tmp_path)
    importer.manual_download_local_file_names = OmegaConf.create([str(tmp_path / "only.zip")])

    assert importer.manual_download_extract_argument() == [str(tmp_path / "only.zip")]


# --- JEDI's consumed inputs (finding 26) ---------------------------------------------


def test_jedi_says_what_is_wrong_when_its_extraction_was_already_consumed(tmp_path):
    """It deletes the nested zips and release dirs as it goes, so an interrupted run
    leaves a payload dir that exists and is empty — and the globs then matched nothing,
    copied nothing, and failed later with an error about a layout that was never wrong."""
    importer = _importer(tmp_path, source="jedi_oceans_cpics")
    extracted = tmp_path / "extracted"
    (extracted / "CPICS_Validated").mkdir(parents=True)
    importer.extracted_dirs = extracted

    with pytest.raises(RuntimeError) as failure:
        importer._prepare_imagefolder()

    assert "force_download=true" in str(failure.value)
