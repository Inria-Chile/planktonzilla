"""
(c) Inria

Network-free tests for the four Tara Pacific importers.

Offline BY CONSTRUCTION: the EcoTaxa client is driven through an injected fake session or
monkeypatched outright, so no test reaches ecotaxa.obs-vlfr.fr. They pin the contract that
makes this source different from every other one in the package:

  * it has NO archive — ``download_uris`` is empty and ``_download_and_extract`` says so
    instead of raising, while the per-object manifests are declared as SIDECARS so the
    pre-flight probes EcoTaxa before a multi-hour build starts;
  * a manifest already on disk is reused, ``force_download`` re-fetches it, and a fetch
    failure names every project, URL and destination;
  * class dirs come from the COMMITTED map keyed by ``classif_id``, so an upstream rename
    is reported and an upstream NEW taxon is skipped rather than silently imported without
    a taxonomy row;
  * unfetchable vignettes are tolerated only up to ``ecotaxa_max_missing_images``;
  * the four sources get four distinct imagefolders (they are four classes for exactly
    that reason).
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import re
from pathlib import Path

import pytest

from planktonzilla.dataset_import import ecotaxa_client, tara_pacific_importer, tara_pacific_layout
from planktonzilla.dataset_import.tara_pacific_importer import (
    TaraPacificBongoDatasetImporter,
    TaraPacificDeckNetDatasetImporter,
    TaraPacificHSNDatasetImporter,
    TaraPacificMantaDatasetImporter,
)

# A tiny stand-in class map: two taxa under each of the two sources the tests exercise, so
# no test depends on the 600-row committed file staying exactly as it is.
FIXTURE_CLASSES = "\n".join(
    [
        "dataset\tclass_dir\tecotaxa_taxon_id",
        "tara_pacific_hsn\tHarosa\t5",
        "tara_pacific_hsn\tCopepoda<Multicrustacea\t25828",
        "tara_pacific_manta\tHarosa\t5",
        "tara_pacific_manta\tfiber<plastic\t85071",
        "",
    ]
)


def _classes_tsv(tmp_path):
    path = tmp_path / "tara_pacific_classes.tsv"
    path.write_text(FIXTURE_CLASSES, encoding="utf-8")
    return path


def _importer(tmp_path, *, projects=(11292,), max_missing=0, cls=TaraPacificHSNDatasetImporter):
    """A real importer wired to the fixture class map and a tmp data dir."""

    class _Fixture(cls):
        CLASSES_TSV = _classes_tsv(tmp_path)

    return _Fixture(
        data_dir=tmp_path,
        human_readable_name="fixture",
        ecotaxa_projects=list(projects),
        ecotaxa_max_missing_images=max_missing,
        show_progress=False,
    )


def _manifest_row(objid, classif_id, display_name, file_name, **overrides):
    row = dict.fromkeys(ecotaxa_client.MANIFEST_COLUMNS)
    row.update(
        objid=objid,
        orig_id=f"orig_{objid}",
        classif_id=classif_id,
        display_name=display_name,
        latitude="33.6",
        longitude="-34.8",
        objdate="2016-06-09",
        objtime="09:40:00",
        depth_min="0.0",
        depth_max="1.0",
        img_file_name=file_name,
    )
    row.update(overrides)
    return row


def _seed_manifest(importer, rows, project_id=11292):
    path = importer.manifests_dir / tara_pacific_importer.manifest_name(project_id)
    return ecotaxa_client.write_manifest(rows, path)


# --- Sidecar protocol -----------------------------------------------------------------


def test_sidecar_targets_are_the_projects_and_the_bundled_class_map(tmp_path):
    importer = _importer(tmp_path, projects=(1344, 1345), cls=TaraPacificMantaDatasetImporter)

    assert importer.sidecar_targets() == [
        ("url", "https://ecotaxa.obs-vlfr.fr/api/projects/1344"),
        ("url", "https://ecotaxa.obs-vlfr.fr/api/projects/1345"),
        ("bundled", str(importer.CLASSES_TSV)),
    ]
    # No archive, so download_targets is exactly the sidecars — and the pre-flight probes
    # EcoTaxa rather than reporting "nothing to fetch".
    assert importer.download_targets() == importer.sidecar_targets()
    assert all(kind in {"url", "bundled"} for kind, _ in importer.download_targets())


def test_missing_sidecars_is_free_and_names_the_manifests(tmp_path):
    importer = _importer(tmp_path)
    assert [p.name for p in importer.missing_sidecars()] == ["ecotaxa_project_11292.tsv"]
    assert importer.missing_sidecars()[0].parent == importer.manifests_dir

    _seed_manifest(importer, [_manifest_row(1, 5, "Harosa", "a/1.jpg")])
    assert importer.missing_sidecars() == []


def test_manifests_dir_is_a_sibling_of_the_imagefolder(tmp_path):
    """Under the RUN's data_dir, like frepj_tables and global_uvp5_aux."""
    importer = _importer(tmp_path)
    assert importer.manifests_dir == tmp_path / tara_pacific_importer.MANIFESTS_DIRNAME
    assert importer.manifests_dir.parent == importer.imagefolder_dir.parent


def test_ensure_sidecars_reuses_a_manifest_on_disk(tmp_path, monkeypatch):
    importer = _importer(tmp_path)
    _seed_manifest(importer, [_manifest_row(1, 5, "Harosa", "a/1.jpg")])

    def _boom(*args, **kwargs):
        raise AssertionError("a manifest already on disk must not be re-fetched")

    monkeypatch.setattr(ecotaxa_client, "fetch_project_manifest", _boom)

    sidecars = importer.ensure_sidecars()
    assert set(sidecars) == {"ecotaxa_project_11292.tsv", "tara_pacific_classes.tsv"}


def test_ensure_sidecars_refetches_under_force_download(tmp_path, monkeypatch):
    """``pz_planktonzilla refresh=redownload`` sets force_download; it must reach here."""
    importer = _importer(tmp_path)
    importer.force_download = True
    _seed_manifest(importer, [_manifest_row(1, 5, "Harosa", "a/1.jpg")])

    calls = []

    def _fake(project_id, **kwargs):
        calls.append(project_id)
        return [_manifest_row(2, 5, "Harosa", "a/2.jpg")]

    monkeypatch.setattr(ecotaxa_client, "fetch_project_manifest", _fake)

    importer.ensure_sidecars()
    assert calls == [11292]
    assert [row["objid"] for row in importer.load_manifest()] == [2]


def test_ensure_sidecars_fetches_what_is_missing(tmp_path, monkeypatch):
    importer = _importer(tmp_path, projects=(1344, 1345), cls=TaraPacificMantaDatasetImporter)
    monkeypatch.setattr(
        ecotaxa_client,
        "fetch_project_manifest",
        lambda project_id, **kwargs: [_manifest_row(project_id * 10, 5, "Harosa", f"a/{project_id}.jpg")],
    )

    sidecars = importer.ensure_sidecars()

    assert sorted(sidecars) == ["ecotaxa_project_1344.tsv", "ecotaxa_project_1345.tsv", "tara_pacific_classes.tsv"]
    assert all(Path(path).exists() for path in sidecars.values())


def test_ensure_sidecars_reports_the_remedy_when_a_fetch_fails(tmp_path, monkeypatch):
    importer = _importer(tmp_path)

    def _fail(project_id, **kwargs):
        raise ecotaxa_client.EcoTaxaError("host down")

    monkeypatch.setattr(ecotaxa_client, "fetch_project_manifest", _fail)

    with pytest.raises(RuntimeError) as excinfo:
        importer.ensure_sidecars()

    message = str(excinfo.value)
    assert "host down" in message
    assert "project 11292" in message
    assert "ecotaxa_project_11292.tsv" in message
    assert "no archive to download by hand" in message


def test_ensure_sidecars_refuses_a_broken_checkout(tmp_path, monkeypatch):
    """The class map ships with the package; no fetch can repair its absence."""
    importer = _importer(tmp_path)
    Path(importer.CLASSES_TSV).unlink()

    with pytest.raises(FileNotFoundError, match="git checkout"):
        importer.ensure_sidecars()


# --- Lifecycle ------------------------------------------------------------------------


def test_download_and_extract_has_nothing_to_fetch(tmp_path):
    """The base class raises on an empty download_uris; this source must not."""
    importer = _importer(tmp_path)
    importer._download_and_extract()

    assert Path(importer.extracted_dirs) == importer.manifests_dir
    assert importer.manifests_dir.is_dir()


def test_load_manifest_reports_the_remedy_when_one_is_absent(tmp_path):
    importer = _importer(tmp_path, projects=(1344, 1345), cls=TaraPacificMantaDatasetImporter)
    _seed_manifest(importer, [_manifest_row(1, 5, "Harosa", "a/1.jpg")], project_id=1344)

    with pytest.raises(FileNotFoundError, match=re.escape("ecotaxa_project_1345.tsv")):
        importer.load_manifest()


def test_load_manifest_concatenates_projects_in_order(tmp_path):
    importer = _importer(tmp_path, projects=(1344, 1345), cls=TaraPacificMantaDatasetImporter)
    _seed_manifest(importer, [_manifest_row(9, 5, "Harosa", "a/9.jpg")], project_id=1344)
    _seed_manifest(importer, [_manifest_row(1, 5, "Harosa", "a/1.jpg")], project_id=1345)

    assert [row["objid"] for row in importer.load_manifest()] == [9, 1]


def test_prepare_imagefolder_lays_out_one_dir_per_class(tmp_path, monkeypatch):
    importer = _importer(tmp_path)
    _seed_manifest(
        importer,
        [
            _manifest_row(1, 5, "Harosa", "a/1.jpg"),
            _manifest_row(2, 25828, "Copepoda<Multicrustacea", "a/2.jpg"),
            _manifest_row(3, 5, "Harosa", "a/3.jpg"),
        ],
    )

    requested = []

    def _fake_download(jobs, **kwargs):
        jobs = list(jobs)
        requested.extend(jobs)
        for _, destination in jobs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"jpeg")
        return len(jobs), 0, []

    monkeypatch.setattr(ecotaxa_client, "download_vault_images", _fake_download)

    importer._prepare_imagefolder()

    assert sorted(p.name for p in importer.imagefolder_dir.iterdir()) == ["Copepoda<Multicrustacea", "Harosa"]
    assert sorted(p.name for p in (importer.imagefolder_dir / "Harosa").iterdir()) == ["1.jpg", "3.jpg"]
    # The file name IS the EcoTaxa object id — the redefiner's only join key back.
    assert [url for url, _ in requested] == [
        "https://ecotaxa.obs-vlfr.fr/vault/a/1.jpg",
        "https://ecotaxa.obs-vlfr.fr/vault/a/2.jpg",
        "https://ecotaxa.obs-vlfr.fr/vault/a/3.jpg",
    ]


def test_prepare_imagefolder_skips_unknown_taxa_and_rows_without_an_image(tmp_path, monkeypatch, caplog):
    """An object with no committed class dir has no Raw_Labels row; importing it would
    land a null taxonomy in the consolidated dataset."""
    importer = _importer(tmp_path)
    _seed_manifest(
        importer,
        [
            _manifest_row(1, 5, "Harosa", "a/1.jpg"),
            _manifest_row(2, 999999, "Brand New Taxon", "a/2.jpg"),
            _manifest_row(3, 5, "Harosa", None),
        ],
    )

    jobs_seen = []

    def _fake_download(jobs, **kwargs):
        jobs = list(jobs)
        jobs_seen.extend(jobs)
        for _, destination in jobs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"jpeg")
        return len(jobs), 0, []

    monkeypatch.setattr(ecotaxa_client, "download_vault_images", _fake_download)

    with caplog.at_level("WARNING"):
        importer._prepare_imagefolder()

    assert [str(destination.name) for _, destination in jobs_seen] == ["1.jpg"]
    assert "999999" in caplog.text and "Brand New Taxon" in caplog.text
    assert "name no image" in caplog.text
    # A class whose objects all dropped out leaves no empty dir behind for the loader.
    assert [p.name for p in importer.imagefolder_dir.iterdir()] == ["Harosa"]


def test_prepare_imagefolder_reports_a_rename_but_keeps_the_committed_spelling(tmp_path, monkeypatch, caplog):
    importer = _importer(tmp_path)
    _seed_manifest(importer, [_manifest_row(1, 25828, "Copepoda<Maxillopoda", "a/1.jpg")])

    def _fake_download(jobs, **kwargs):
        jobs = list(jobs)
        for _, destination in jobs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"jpeg")
        return len(jobs), 0, []

    monkeypatch.setattr(ecotaxa_client, "download_vault_images", _fake_download)

    with caplog.at_level("WARNING"):
        importer._prepare_imagefolder()

    assert "renamed" in caplog.text
    assert (importer.imagefolder_dir / "Copepoda<Multicrustacea" / "1.jpg").exists()
    assert not (importer.imagefolder_dir / "Copepoda<Maxillopoda").exists()


def test_prepare_imagefolder_refuses_to_finish_with_missing_vignettes(tmp_path, monkeypatch):
    importer = _importer(tmp_path, max_missing=0)
    _seed_manifest(importer, [_manifest_row(1, 5, "Harosa", "a/1.jpg")])

    monkeypatch.setattr(ecotaxa_client, "download_vault_images", lambda jobs, **kwargs: (0, 0, ["a/1.jpg: HTTP 500"]))

    with pytest.raises(RuntimeError) as excinfo:
        importer._prepare_imagefolder()

    assert "Re-run to resume" in str(excinfo.value)
    assert "ecotaxa_max_missing_images" in str(excinfo.value)


def test_prepare_imagefolder_accepts_a_tolerated_remainder(tmp_path, monkeypatch):
    importer = _importer(tmp_path, max_missing=1)
    _seed_manifest(importer, [_manifest_row(1, 5, "Harosa", "a/1.jpg")])

    monkeypatch.setattr(ecotaxa_client, "download_vault_images", lambda jobs, **kwargs: (0, 0, ["a/1.jpg: HTTP 500"]))

    importer._prepare_imagefolder()  # does not raise


# --- The four sources -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "source_name"),
    [
        (TaraPacificBongoDatasetImporter, "tara_pacific_bongo"),
        (TaraPacificDeckNetDatasetImporter, "tara_pacific_decknet"),
        (TaraPacificHSNDatasetImporter, "tara_pacific_hsn"),
        (TaraPacificMantaDatasetImporter, "tara_pacific_manta"),
    ],
)
def test_each_source_defaults_to_its_recorded_projects(cls, source_name, tmp_path):
    importer = cls(data_dir=tmp_path, show_progress=False)

    assert importer.SOURCE_NAME == source_name
    assert importer.projects == tara_pacific_layout.SOURCES[source_name]["projects"]
    assert len(importer.class_map) == len(tara_pacific_layout.load_class_map(source_name))


def test_the_four_sources_get_four_distinct_imagefolders(tmp_path):
    """They are four classes precisely because imagefolder_dir is derived from the class
    name: one shared class would make four configs overwrite one directory."""
    classes = (
        TaraPacificBongoDatasetImporter,
        TaraPacificDeckNetDatasetImporter,
        TaraPacificHSNDatasetImporter,
        TaraPacificMantaDatasetImporter,
    )
    folders = {cls(data_dir=tmp_path, show_progress=False).imagefolder_dir for cls in classes}
    assert len(folders) == 4
    # ...while the manifests are SHARED, because the seven projects are disjoint.
    manifest_dirs = {cls(data_dir=tmp_path, show_progress=False).manifests_dir for cls in classes}
    assert len(manifest_dirs) == 1


def test_a_subclass_without_a_source_name_is_rejected(tmp_path):
    class _Nameless(tara_pacific_importer.TaraPacificDatasetImporter):
        pass

    with pytest.raises(ValueError, match="SOURCE_NAME"):
        _Nameless(data_dir=tmp_path, show_progress=False)
