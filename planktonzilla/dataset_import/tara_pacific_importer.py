"""
(c) Inria

Tara Pacific importers: build an imagefolder from the public EcoTaxa projects that hold
the images of Mériguet et al. (2025, ESSD 17, 2761-2792).

Why this source has no archive
------------------------------
Every other importer in this package downloads an archive and rearranges it. The four
SEANOE deposits this milestone answers (10.17882/102694, /102697, /102336, /102537)
publish EcoTaxa **TSV exports** — per-object metadata, no vignettes — and say so
themselves: "All images and their taxonomic annotations are available in the open-access
EcoTaxa project at these links". EcoTaxa's own archive export needs an account
(``POST /api/object_set/export`` answers ``403 Not authenticated`` anonymously), so the
only anonymous route to the images is the pair of public read endpoints wrapped by
:mod:`planktonzilla.dataset_import.ecotaxa_client`.

So the lifecycle is re-seated rather than re-implemented:

  * ``_download_and_extract`` has NOTHING to fetch and says so instead of raising on an
    empty ``download_uris``;
  * the per-object MANIFEST is a SIDECAR — the base class's name for a build-time input
    outside the archive lifecycle that every run needs, imagefolder reused or not. It is
    exactly that: ``_prepare_imagefolder`` reads it to lay out the images, and
    :class:`~planktonzilla.planktonzilla_dataset.generate_planktonzilla.TaraPacificRedefiner`
    reads it again, on every run, for each object's coordinates, date and depth. Fetching
    it through ``ensure_sidecars`` also means the pre-flight probes EcoTaxa
    (``pz_planktonzilla dry_run=true`` reports it under ``sidecars:``) instead of
    discovering a dead host hours in;
  * ``_prepare_imagefolder`` fetches one vignette per object into
    ``<class dir>/<objid>.jpg``, resumably.

Class dirs are named from the COMMITTED ``tara_pacific_classes.tsv`` map keyed by the
manifest's ``classif_id``, never from the live ``display_name`` — see
:mod:`planktonzilla.dataset_import.tara_pacific_layout` for why (EcoTaxa renames taxa in
place, and a renamed class dir would silently repoint every ``Raw_Labels`` join key).
"""

from pathlib import Path

import requests

from planktonzilla.dataset_import import ecotaxa_client, tara_pacific_layout
from planktonzilla.dataset_import.dataset_importer import DatasetImporter, is_dir_empty
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# Sibling of raw_dir / imagefolder_dir under the RUN's data_dir, following the
# `global_uvp5_aux` / `frepj_tables` idiom. One directory for all four sources: the seven
# EcoTaxa projects are disjoint, so one file per project can never be claimed by two
# sources, and a machine that builds several of them keeps ONE copy of each manifest.
MANIFESTS_DIRNAME = "tara_pacific_manifests"


def manifest_name(project_id: int) -> str:
    """The manifest file name for one EcoTaxa project."""
    return f"ecotaxa_project_{project_id}.tsv"


def manifest_paths(manifests_dir: str | Path, project_ids) -> list[Path]:
    """Where one source's manifests live, in project order."""
    manifests_dir = Path(manifests_dir)
    return [manifests_dir / manifest_name(project_id) for project_id in project_ids]


class TaraPacificDatasetImporter(DatasetImporter):
    """Base importer for one Tara Pacific source, assembled from public EcoTaxa projects.

    Concrete subclasses set :attr:`SOURCE_NAME` only. They exist as separate classes
    rather than one class configured four ways because ``DatasetImporter.__post_init__``
    derives ``imagefolder_dir`` and ``raw_dir`` from the CLASS name: four configs sharing
    one class would share one imagefolder and overwrite each other.

    ``ecotaxa_projects`` defaults to the project ids recorded in
    :data:`tara_pacific_layout.SOURCES`; a config (or a test) may override it.
    """

    #: Key into :data:`tara_pacific_layout.SOURCES`; set by every concrete subclass.
    SOURCE_NAME = ""

    #: Overridden by tests to point the class map at a fixture.
    CLASSES_TSV = tara_pacific_layout.CLASSES_TSV

    def __post_init__(self):
        super().__post_init__()
        if not self.SOURCE_NAME:
            raise ValueError(f"{type(self).__name__} must set SOURCE_NAME; it names the source in tara_pacific_layout.")
        self.manifests_dir = self.data_dir / MANIFESTS_DIRNAME
        # `or` (not `is None`): an empty list is the same "nothing configured" as an unset
        # field, and importing a source with no project would silently produce nothing.
        self.projects = tuple(
            int(p) for p in (self.ecotaxa_projects or tara_pacific_layout.SOURCES[self.SOURCE_NAME]["projects"])
        )
        self.class_map = tara_pacific_layout.load_class_map(self.SOURCE_NAME, self.CLASSES_TSV)

    # --- Sidecars: the per-object manifests, needed on EVERY run ----------------------

    def sidecar_targets(self) -> list[tuple[str, str]]:
        """The EcoTaxa project endpoints (fetched) and the committed class map (bundled)."""
        return [("url", ecotaxa_client.project_url(project_id)) for project_id in self.projects] + [
            ("bundled", str(self.CLASSES_TSV))
        ]

    def missing_sidecars(self) -> list[Path]:
        """Manifests not on disk — what a run would fetch from EcoTaxa."""
        return [path for path in manifest_paths(self.manifests_dir, self.projects) if not path.exists()]

    def ensure_sidecars(self) -> dict[str, Path]:
        """Fetch the missing manifests into ``manifests_dir``; return every sidecar path.

        The committed class map is checked first: it is not downloadable, so its absence is
        a broken checkout that no fetch can repair. A manifest already on disk is REUSED —
        re-paging a million objects to learn what a file already says is pure cost — unless
        ``force_download`` is set, which is what ``pz_planktonzilla refresh=redownload``
        passes down.

        Raises:
            FileNotFoundError: If the committed class map is missing.
            RuntimeError: If a manifest cannot be fetched, naming every project, its URL
                and its destination.
        """
        classes_tsv = Path(self.CLASSES_TSV)
        if not classes_tsv.exists():
            raise FileNotFoundError(
                f"«{self.SOURCE_NAME}» needs the committed class map {classes_tsv}, which is not on disk. It ships "
                f"with the package: restore it with `git checkout -- {classes_tsv}`."
            )

        sidecars: dict[str, Path] = {classes_tsv.name: classes_tsv}
        with requests.Session() as session:
            for project_id, path in zip(self.projects, manifest_paths(self.manifests_dir, self.projects)):
                if path.exists() and not self.force_download:
                    logger.info(f"Reusing EcoTaxa manifest «{path}» ({project_id}).")
                    sidecars[path.name] = path
                    continue
                logger.info(f"Fetching the EcoTaxa manifest of project {project_id} into «{path}».")
                try:
                    rows = ecotaxa_client.fetch_project_manifest(
                        project_id,
                        session=session,
                        user_agent=self.http_user_agent,
                        window_size=self.ecotaxa_window_size,
                        timeout=self.http_timeout,
                        retries=self.max_download_retries,
                        show_progress=self.show_progress,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"«{self.SOURCE_NAME}» could not obtain the EcoTaxa manifest of project {project_id}: "
                        f"{type(e).__name__}: {e}\n" + self.sidecar_instructions()
                    ) from e
                ecotaxa_client.write_manifest(rows, path)
                sidecars[path.name] = path
        return sidecars

    def sidecar_instructions(self) -> str:
        """What a human needs in order to unblock a failed manifest fetch."""
        lines = [
            f"«{self.human_readable_name or self.SOURCE_NAME}» reads its per-object manifest from the public EcoTaxa "
            "API — there is no archive to download by hand, because the SEANOE deposit holds metadata only:"
        ]
        for project_id, path in zip(self.projects, manifest_paths(self.manifests_dir, self.projects)):
            lines.append(f"  - project {project_id}: {ecotaxa_client.query_url(project_id)} -> {path}")
        lines.append(
            "Check https://ecotaxa.obs-vlfr.fr is reachable and that the projects are still public, then re-run. "
            "A manifest already on disk is reused; delete it to force a re-fetch."
        )
        return "\n".join(lines)

    # --- Lifecycle --------------------------------------------------------------------

    def _download_and_extract(self):
        """Nothing to download: this source has no archive.

        The base implementation raises when ``download_uris`` is empty, which is the right
        answer for an archive-backed source and the wrong one here. ``extracted_dirs`` is
        pointed at the manifests directory so the base class's "extraction failed" guard
        sees a real path, and so a reader of a log can tell what stood in for the archive.
        """
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dirs = str(self.manifests_dir)
        logger.info(
            f"«{self.SOURCE_NAME}» has no archive: its images come one per object from the EcoTaxa vault, "
            f"driven by the manifests in «{self.manifests_dir}»."
        )

    def load_manifest(self) -> list[dict]:
        """Every object of this source, in project order then object-id order.

        Raises:
            FileNotFoundError: If a manifest is absent — ``ensure_sidecars`` obtains them,
                and it runs before this in both entry points.
        """
        rows: list[dict] = []
        for project_id, path in zip(self.projects, manifest_paths(self.manifests_dir, self.projects)):
            if not path.exists():
                raise FileNotFoundError(
                    f"«{self.SOURCE_NAME}» is missing the manifest of EcoTaxa project {project_id} at «{path}».\n"
                    + self.sidecar_instructions()
                )
            rows.extend(ecotaxa_client.read_manifest(path))
        return rows

    def expected_image_count(self) -> int:
        """How many vignettes a finished imagefolder holds, per the manifest.

        Counts exactly the rows :meth:`_prepare_imagefolder` turns into a fetch: an object
        whose taxon is in the committed class map AND that carries an image. Rows it skips
        are excluded, so a complete import really does reach this number.
        """
        rows = self.load_manifest()
        return sum(1 for row in rows if self.class_map.get(row["classif_id"]) is not None and row["img_file_name"])

    def imagefolder_is_complete(self) -> bool:
        """True only when the imagefolder holds every vignette the manifest names.

        The base class answers "is the directory non-empty?", which is the right question
        for a source whose imagefolder is written in one pass out of an extracted archive.
        It is the WRONG question here. This imagefolder is filled one HTTP fetch at a time
        over hours, so an interrupted run leaves it non-empty and PARTIAL — and under the
        inherited answer the pipeline reused that fraction as though it were the finished
        source, carrying it into the consolidated dataset (and, with ``push_to_hub``, onto
        the Hub) labelled as the whole deposit.

        So the question is asked with a count instead. Both sides are cheap relative to the
        fetch they gate: the manifest is a local TSV read, and the images are a directory
        walk. ``ecotaxa_max_missing_images`` is honoured, so vignettes that are permanently
        gone upstream do not make every future run re-walk the whole source.

        Never raises: a missing manifest (the state before ``ensure_sidecars`` has run)
        means "not complete", which is the answer that makes the caller build.
        """
        if is_dir_empty(self.imagefolder_dir):
            return False
        try:
            expected = self.expected_image_count()
        except (FileNotFoundError, ValueError) as e:
            logger.info(f"«{self.SOURCE_NAME}» cannot count its manifest yet ({type(e).__name__}); treating as incomplete.")
            return False

        present = sum(1 for _ in self.imagefolder_dir.rglob(f"*{tara_pacific_layout.IMAGE_SUFFIX}"))
        if present + self.ecotaxa_max_missing_images >= expected:
            return True

        logger.warning(
            f"«{self.SOURCE_NAME}» imagefolder «{self.imagefolder_dir}» holds {present} of the {expected} vignette(s) "
            f"its manifest names — an earlier run was interrupted. Resuming the fetch; nothing already on disk is "
            "re-downloaded. Raise dataset_import.ecotaxa_max_missing_images if the remainder is permanently gone "
            "upstream."
        )
        return False

    def _prepare_imagefolder(self):
        """Fetch one vignette per manifest row into ``<class dir>/<objid>.jpg``.

        Resumable: a vignette already on disk is not re-fetched, so an interrupted run is
        finished by re-running rather than restarted. Rows whose ``classif_id`` is not in
        the committed class map, or that carry no image, are counted and reported but never
        guessed at — an object with no ``Raw_Labels`` row would land in the consolidated
        dataset with a null taxonomy.

        Raises:
            RuntimeError: If more than ``ecotaxa_max_missing_images`` vignettes could not
                be fetched after their retries. The remedy is to re-run: the fetch resumes.
        """
        rows = self.load_manifest()
        logger.info(f"«{self.SOURCE_NAME}» manifest: {len(rows)} object(s) across project(s) {list(self.projects)}.")

        renamed, unknown = tara_pacific_layout.reconcile_display_names(rows, self.class_map)
        if renamed:
            shown = ", ".join(f"{taxon}: «{self.class_map[taxon]}» -> «{live}»" for taxon, live in sorted(renamed.items())[:5])
            logger.warning(
                f"EcoTaxa has renamed {len(renamed)} taxon/taxa since {Path(self.CLASSES_TSV).name} was frozen "
                f"({shown}). The committed spelling is kept, so no Raw_Labels join key moves."
            )

        jobs, skipped_unknown, skipped_no_image = [], 0, 0
        for row in rows:
            class_dir = self.class_map.get(row["classif_id"])
            if class_dir is None:
                skipped_unknown += 1
                continue
            file_name = row["img_file_name"]
            if not file_name:
                skipped_no_image += 1
                continue
            destination = self.imagefolder_dir / class_dir / tara_pacific_layout.image_file_name(row["objid"])
            jobs.append((ecotaxa_client.vault_url(file_name), destination))

        if skipped_unknown:
            shown = ", ".join(f"{taxon} («{name}»)" for taxon, name in sorted(unknown.items())[:5])
            logger.warning(
                f"{skipped_unknown} object(s) carry {len(unknown)} EcoTaxa taxon/taxa absent from "
                f"{Path(self.CLASSES_TSV).name} ({shown}); they are NOT imported, because they have no taxonomy row. "
                "Add them to the class map and to planktonzilla_taxonomy.csv to include them."
            )
        if skipped_no_image:
            logger.warning(f"{skipped_no_image} manifest row(s) name no image and are skipped.")

        # One directory per class up front, even for a class whose objects all failed:
        # the class set is then a function of the committed map, not of what the network
        # happened to serve. Empty ones are removed below so the loader never sees them.
        for class_dir in sorted(set(self.class_map.values())):
            (self.imagefolder_dir / class_dir).mkdir(parents=True, exist_ok=True)

        with requests.Session() as session:
            fetched, skipped, failures = ecotaxa_client.download_vault_images(
                jobs,
                session=session,
                user_agent=self.http_user_agent,
                workers=self.ecotaxa_image_workers,
                timeout=self.http_timeout,
                retries=self.max_download_retries,
                show_progress=self.show_progress,
                desc=f"{self.SOURCE_NAME} vignettes",
            )

        logger.info(
            f"«{self.SOURCE_NAME}»: {fetched} vignette(s) fetched, {skipped} already on disk, "
            f"{len(failures)} failed, of {len(jobs)} in «{self.imagefolder_dir}»."
        )

        for class_dir in sorted(set(self.class_map.values())):
            path = self.imagefolder_dir / class_dir
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

        if len(failures) > self.ecotaxa_max_missing_images:
            head = "\n".join(f"  - {failure}" for failure in failures[:5])
            raise RuntimeError(
                f"«{self.SOURCE_NAME}»: {len(failures)} vignette(s) could not be fetched, above the tolerated "
                f"{self.ecotaxa_max_missing_images}. First failures:\n{head}\n"
                "Re-run to resume — every vignette already on disk is skipped. Raise "
                "dataset_import.ecotaxa_max_missing_images to accept a persistent remainder."
            )


class TaraPacificBongoDatasetImporter(TaraPacificDatasetImporter):
    """Bongo net + FlowCam micro-plankton (SEANOE 10.17882/102694; EcoTaxa 11370, 11369)."""

    SOURCE_NAME = "tara_pacific_bongo"


class TaraPacificDeckNetDatasetImporter(TaraPacificDatasetImporter):
    """Deck net + FlowCam micro-plankton (SEANOE 10.17882/102697; EcoTaxa 11353, 11341).

    The largest of the four (1.58 M objects) and the one whose SEANOE archive is unusable:
    ``.../102697/data/114288.zip`` is served complete but is internally corrupt (its
    central directory overshoots the file by exactly 4 000 000 bytes). Nothing here reads
    it — the images come from EcoTaxa either way.
    """

    SOURCE_NAME = "tara_pacific_decknet"


class TaraPacificHSNDatasetImporter(TaraPacificDatasetImporter):
    """High-Speed Net + ZooScan meso-plankton (SEANOE 10.17882/102336; EcoTaxa 11292)."""

    SOURCE_NAME = "tara_pacific_hsn"


class TaraPacificMantaDatasetImporter(TaraPacificDatasetImporter):
    """Manta net + ZooScan meso-plankton AND microplastics (SEANOE 10.17882/102537).

    The only source in the registry whose label space includes anthropogenic microplastic
    classes (``fiber<plastic``, ``film``, ``fragment``, ``pellet``, ``polystyrene``),
    which come from the separate EcoTaxa plastics project 1345 alongside the plankton
    project 1344.
    """

    SOURCE_NAME = "tara_pacific_manta"
