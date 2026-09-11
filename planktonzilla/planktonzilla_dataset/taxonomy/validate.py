"""
(c) Inria

The enforcer: executes ``data/datapackage.json`` against the package, plus the rules no table
schema can express.

Step 3 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``. The plan's shape was "a Frictionless
descriptor as the declarative schema, and a polars module that re-implements it so a laptop
without Frictionless gives CI's verdict". Re-implementation is a drift hazard — two statements of
one constraint, and nothing making them agree — so this module **executes the descriptor
directly** instead. There is one execution path; the descriptor is the schema of record rather
than documentation beside the enforcer, and a constraint declared there cannot be silently
unenforced. Frictionless remains an optional cross-check on the same file, not a requirement.
(Open decision §10.5, resolved that way for a further reason: the package could not be installed
in this environment at all, and an unverifiable dev dependency is not a gate.)

Four rules exist because tools the design trusted do NOT enforce them, each measured by the
panel and each with its own failing fixture:

    (scientificName, taxonRank, parentNameUsageID) unique   Frictionless 5.19 parses `uniqueKeys`
                                                            and reports an injected duplicate valid
    one `exact` id per (concept, published authority)        a second id is schema-legal in any
                                                            SSSOM table and vanishes by sort order
    datasetID == mapping file stem                           a foreign row passes a per-file key
    the absence row must actually validate                   the design's own pattern rejects the
                                                            row it offers for "not in this register"

Findings are keyed by a hash of their own VALUES, never by row position or row count. That is the
discipline ``utils/verify_label_consistency.py`` already uses and the one thing in the repository
where adding a source does not invalidate a curated judgement: a waiver survives a new source
adopting a label whose taxon is already known, and lapses exactly when the finding itself changes.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from planktonzilla.planktonzilla_dataset.taxonomy.model import BROAD_MATCH, EXACT_MATCH, TaxonomyError, read_tsv

SEVERITY_ERROR = "ERROR"
SEVERITY_WARN = "WARN"


@dataclass(frozen=True)
class Finding:
    """One rule violation, identified by what it says rather than by where it was found."""

    check: str
    severity: str
    resource: str
    locator: str
    detail: str

    @property
    def finding_id(self) -> str:
        """A stable 12-hex digest of the finding's own values.

        Never the row number, never the row count: a manifest that renumbers, or a source that
        appends, must not silently retire somebody's adjudication.
        """
        payload = "|".join((self.check, self.resource, self.locator, self.detail))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> str:
        return f"[{self.severity}] {self.finding_id} {self.check} {self.resource}:{self.locator} — {self.detail}"


@dataclass
class Report:
    """Every finding, and the waivers that did or did not match one."""

    findings: list = field(default_factory=list)
    waived: list = field(default_factory=list)
    stale_waivers: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [f for f in self.findings if f.severity == SEVERITY_ERROR]

    def summary(self) -> dict:
        by_check = {}
        for finding in self.findings:
            by_check[finding.check] = by_check.get(finding.check, 0) + 1
        return {
            "total": len(self.findings),
            "errors": len(self.errors),
            "waived": len(self.waived),
            "stale_waivers": len(self.stale_waivers),
            "by_check": dict(sorted(by_check.items())),
        }


def load_descriptor(package_dir: Path) -> dict:
    """Read ``datapackage.json``. Plain JSON on purpose: the schema of record needs no dependency."""
    path = Path(package_dir) / "datapackage.json"
    if not path.exists():
        raise TaxonomyError(f"no descriptor at «{path}»")
    return json.loads(path.read_text(encoding="utf-8"))


def _resource_paths(package_dir: Path, resource: dict) -> list:
    """The files one resource covers: an explicit ``path``, or the files a ``pathGlob`` matches.

    ``pathGlob`` is a project extension, used only by the mapping resource. It exists so the
    descriptor declares the SHAPE of a mapping file and nothing about which sources exist: an
    enumeration here would be a second source registry beside ``constants.DATASET_IMPORT_CONFIGS``,
    and a line that both branches of a concurrent add-source PR insert at the same position.
    """
    package_dir = Path(package_dir)
    if "pathGlob" in resource:
        return sorted(package_dir.glob(resource["pathGlob"]))
    paths = resource["path"]
    return [package_dir / p for p in ([paths] if isinstance(paths, str) else paths)]


def _vocabulary_values(package_dir: Path, descriptor: dict, reference: str) -> set:
    """Resolve a ``vocabulary`` constraint like ``rank#rank`` to the terms in that table."""
    resource_name, _, column = reference.partition("#")
    resource = next((r for r in descriptor["resources"] if r["name"] == resource_name), None)
    if resource is None:
        raise TaxonomyError(f"vocabulary {reference!r} names no resource")
    values = set()
    for path in _resource_paths(package_dir, resource):
        values |= {row[column] for row in read_tsv(path)}
    return values


# Executing the descriptor
def check_descriptor(package_dir: Path, descriptor: dict) -> list:
    """Every constraint the descriptor declares: fields, required, pattern, enum, vocabulary, keys."""
    package_dir = Path(package_dir)
    findings = []
    by_name = {resource["name"]: resource for resource in descriptor["resources"]}
    rows_by_resource = {}

    for resource in descriptor["resources"]:
        name = resource["name"]
        schema = resource["schema"]
        declared = [f["name"] for f in schema["fields"]]
        collected = []

        for path in _resource_paths(package_dir, resource):
            if not path.exists():
                findings.append(
                    Finding("missing_file", SEVERITY_ERROR, name, str(path), "declared by the descriptor, absent on disk")
                )
                continue
            rows = read_tsv(path)
            if rows and list(rows[0]) != declared:
                findings.append(
                    Finding(
                        "header_mismatch", SEVERITY_ERROR, name, path.name, f"columns {list(rows[0])} != declared {declared}"
                    )
                )
                continue
            for row in rows:
                row["__file__"] = path.name
            collected.extend(rows)

        rows_by_resource[name] = collected

        for spec in schema["fields"]:
            constraints = spec.get("constraints", {})
            if not constraints:
                continue
            column = spec["name"]
            pattern = re.compile(constraints["pattern"]) if "pattern" in constraints else None
            allowed = set(constraints["enum"]) if "enum" in constraints else None
            vocabulary = (
                _vocabulary_values(package_dir, descriptor, constraints["vocabulary"]) if "vocabulary" in constraints else None
            )
            for row in collected:
                value = row[column]
                locator = f"{row['__file__']}:{column}"
                if constraints.get("required") and value == "":
                    findings.append(Finding("required", SEVERITY_ERROR, name, locator, f"{column} is empty"))
                    continue
                if pattern is not None and not pattern.match(value):
                    findings.append(
                        Finding("pattern", SEVERITY_ERROR, name, locator, f"{value!r} does not match {constraints['pattern']}")
                    )
                # An OPTIONAL field left empty is an absent value, not a vocabulary violation. The
                # three patterns in this descriptor say so themselves (`^(pzt:[0-9]{6})?$`), which
                # an enum has no way to express — so the rule lives here instead. `method` is the
                # first optional vocabulary field: 1529 rows have no recorded provenance, and that
                # is a gap to close rather than 1529 errors.
                optional_blank = value == "" and not constraints.get("required")
                if allowed is not None and not optional_blank and value not in allowed:
                    findings.append(Finding("enum", SEVERITY_ERROR, name, locator, f"{value!r} not in {sorted(allowed)}"))
                if vocabulary is not None and not optional_blank and value not in vocabulary:
                    findings.append(
                        Finding("vocabulary", SEVERITY_ERROR, name, locator, f"{value!r} not in {constraints['vocabulary']}")
                    )

        findings.extend(_check_keys(name, schema, collected))

    findings.extend(_check_foreign_keys(descriptor, by_name, rows_by_resource))
    return findings


def _check_keys(name: str, schema: dict, rows) -> list:
    """Primary key and, crucially, ``uniqueKeys`` — which Frictionless parses but does not enforce."""
    findings = []
    for label, columns in [("primary_key", schema.get("primaryKey"))] + [
        ("unique_key", key) for key in schema.get("uniqueKeys", [])
    ]:
        if not columns:
            continue
        seen = {}
        for row in rows:
            key = tuple(row[column] for column in columns)
            seen.setdefault(key, []).append(row["__file__"])
        for key, files in sorted(seen.items()):
            if len(files) > 1:
                findings.append(
                    Finding(
                        label,
                        SEVERITY_ERROR,
                        name,
                        "+".join(sorted(set(files))),
                        f"{list(columns)} = {list(key)} appears {len(files)} times",
                    )
                )
    return findings


def _check_foreign_keys(descriptor: dict, by_name: dict, rows_by_resource: dict) -> list:
    findings = []
    for resource in descriptor["resources"]:
        for foreign in resource["schema"].get("foreignKeys", []):
            reference = foreign["reference"]
            targets = {
                tuple(row[column] for column in reference["fields"]) for row in rows_by_resource.get(reference["resource"], [])
            }
            for row in rows_by_resource.get(resource["name"], []):
                key = tuple(row[column] for column in foreign["fields"])
                if all(value == "" for value in key) or key in targets:
                    continue
                findings.append(
                    Finding(
                        "foreign_key",
                        SEVERITY_ERROR,
                        resource["name"],
                        row["__file__"],
                        f"{foreign['fields']} = {list(key)} resolves to no {reference['resource']}",
                    )
                )
    return findings


# Rules no table schema can express
def check_rules(package_dir: Path, descriptor: dict, registered=None) -> list:
    """The structural and semantic rules a Table Schema cannot state."""
    package_dir = Path(package_dir)
    findings = []
    taxa = {row["taxonID"]: row for row in read_tsv(package_dir / "taxon.tsv")}
    authorities = {row["prefix"]: row for row in read_tsv(package_dir / "vocab" / "authority.tsv")}
    legacy_slot = {row["rank"]: row["legacy_slot"] for row in read_tsv(package_dir / "vocab" / "rank.tsv")}

    findings += _check_no_cycles(taxa)
    findings += _check_lineage_name_repeat(taxa, legacy_slot)
    findings += _check_single_valued_authorities(package_dir, authorities)
    findings += _check_shared_ids(package_dir, taxa, authorities)
    findings += _check_mapping_files(package_dir, descriptor, taxa, registered)
    findings += _check_canonical_names(taxa)
    return findings


def _check_no_cycles(taxa: dict) -> list:
    findings = []
    for taxon_id in taxa:
        seen, current = set(), taxon_id
        while current:
            if current in seen:
                findings.append(Finding("parent_cycle", SEVERITY_ERROR, "taxon", taxon_id, f"cycle through {current}"))
                break
            seen.add(current)
            current = taxa[current]["parentNameUsageID"] if current in taxa else ""
    return findings


def _check_lineage_name_repeat(taxa: dict, legacy_slot: dict) -> list:
    """A name must not repeat within its own lineage — scoped to ranks that HAVE a legacy slot.

    The scoping is the point. Unscoped, a ``bacteria`` domain node above the ``bacteria`` kingdom
    fires 30 times on data that is entirely correct, which is how a rule like this gets disabled.
    """
    findings = []
    for taxon_id, row in sorted(taxa.items()):
        if not legacy_slot.get(row["taxonRank"], ""):
            continue
        # Only the node that INTRODUCES the repeat. Reporting every descendant as well turned four
        # real cases into sixteen findings, twelve of which named an innocent node.
        ancestors, current = [], row["parentNameUsageID"]
        while current and current in taxa:
            node = taxa[current]
            if legacy_slot.get(node["taxonRank"], ""):
                ancestors.append(node["scientificName"])
            current = node["parentNameUsageID"]
        if row["scientificName"] in ancestors:
            findings.append(
                Finding(
                    "lineage_name_repeat",
                    SEVERITY_WARN,
                    "taxon",
                    taxon_id,
                    f"{row['scientificName']!r} already appears higher in its own lineage",
                )
            )
    return findings


def _check_single_valued_authorities(package_dir: Path, authorities: dict) -> list:
    """Exactly one ``exact`` id per (concept, single-valued authority).

    A second is schema-legal in any SSSOM-shaped table and would be dropped by the renderer with
    the survivor decided by numeric sort order rather than by the model.
    """
    counts = {}
    for row in read_tsv(package_dir / "identifier.tsv"):
        if row["predicate_id"] != EXACT_MATCH:
            continue
        prefix = row["object_id"].partition(":")[0]
        if authorities.get(prefix, {}).get("multi_valued") == "true":
            continue
        counts.setdefault((row["subject_id"], prefix), []).append(row["object_id"])

    return [
        Finding(
            "multiple_exact_ids",
            SEVERITY_ERROR,
            "identifier",
            subject,
            f"{prefix} is single-valued but carries {sorted(objects)}",
        )
        for (subject, prefix), objects in sorted(counts.items())
        if len(objects) > 1
    ]


def _check_shared_ids(package_dir: Path, taxa: dict, authorities: dict) -> list:
    """One authority id claimed by two concepts, split into the two cases that mean different things.

    Undifferentiated this fires 296 times, almost all of them the same benign shape, which is how a
    rule gets switched off. Two separations make it readable.

    ``ecotaxa_legacy`` is excluded outright: it is a multi-valued legacy space where sharing carries
    no claim at all.

    The rest split by whether one claimant is an ANCESTOR of the others. A species carrying its own
    genus's id is a *coarse* identifier, not a collision — the register had nothing finer. It gets
    its own name because it is a backlog with a known remedy — a ``skos:broadMatch`` row, which step
    7 seeded for all 147 of them — and because burying it with the real collisions hides them. An id
    shared by concepts on different branches is the KI-13 shape and stays a collision.

    Since step 7 the first loop below checks the seeding itself: a broad match is only honest if
    some exact-match holder of the same id is an ancestor of the subject. Without that, relabelling
    a cross-branch collision as a broad match would make it disappear from this report.
    """

    def ancestry(taxon_id):
        chain, current = set(), taxon_id
        while current and current in taxa:
            chain.add(current)
            current = taxa[current]["parentNameUsageID"]
        return chain

    owners = {}
    for row in read_tsv(package_dir / "identifier.tsv"):
        prefix = row["object_id"].partition(":")[0]
        if row["predicate_id"] != EXACT_MATCH or row["object_id"].endswith(":absent"):
            continue
        if authorities.get(prefix, {}).get("multi_valued") == "true":
            continue
        owners.setdefault(row["object_id"], set()).add(row["subject_id"])

    # Step 7 re-predicated the coarse identifiers, which is what stops them being reported here —
    # so the honesty of that re-predication has to be checked, or "call it a broad match" becomes a
    # way to silence a real collision. A broad match claims the subject is NARROWER than the id's
    # concept, which is only true if some exact-match holder of that id is one of its ancestors.
    broad = {}
    for row in read_tsv(package_dir / "identifier.tsv"):
        if row["predicate_id"] == BROAD_MATCH:
            broad.setdefault(row["object_id"], set()).add(row["subject_id"])

    findings = [
        Finding(
            "unjustified_broad_match",
            SEVERITY_ERROR,
            "identifier",
            f"{subject} {object_id}",
            "claims to be narrower than this id, but no exact-match holder of it is an ancestor",
        )
        for object_id, subjects in sorted(broad.items())
        for subject in sorted(subjects)
        if not any(holder in ancestry(subject) for holder in owners.get(object_id, set()))
    ]

    for object_id, subjects in sorted(owners.items()):
        if len(subjects) < 2:
            continue
        shallowest = min(subjects, key=lambda t: len(ancestry(t)))
        if all(shallowest in ancestry(subject) for subject in subjects):
            findings.append(
                Finding(
                    "coarse_identifier",
                    SEVERITY_WARN,
                    "identifier",
                    object_id,
                    f"{len(subjects)} concepts under {taxa[shallowest]['scientificName']!r} share it; "
                    f"the finer ones want skos:broadMatch",
                )
            )
        else:
            findings.append(
                Finding(
                    "id_shared_across_branches",
                    SEVERITY_WARN,
                    "identifier",
                    object_id,
                    f"claimed by unrelated concepts {sorted(subjects)}",
                )
            )
    return findings


def _check_mapping_files(package_dir: Path, descriptor: dict, taxa: dict, registered=None) -> list:
    """``datasetID == file stem``, the display-name agreement, and registry/disk agreement.

    A row for one source dropped into another's file passes a per-file primary key unnoticed, and
    the exporter then keeps whichever file the glob visits last.
    """
    findings = []
    resource = next(r for r in descriptor["resources"] if r["name"] == "mapping")
    # Disk against the ONE registry, not against a second copy in the descriptor. `registered` is
    # constants.DATASET_IMPORT_CONFIGS, which adding a source already has to touch anyway for
    # validate_license_coverage; checking against a descriptor list as well would be two lists to
    # keep in step. `registered=None` means the caller did not offer a registry, so the direction
    # that needs one is skipped rather than guessed at.
    on_disk = {path.stem for path in _resource_paths(package_dir, resource)}
    if registered is not None:
        findings += [
            Finding("registry_drift", SEVERITY_ERROR, "mapping", name, "registered, but no mapping file on disk")
            for name in sorted(set(registered) - on_disk)
        ]
        findings += [
            Finding("registry_drift", SEVERITY_ERROR, "mapping", name, "a mapping file on disk, but not a registered source")
            for name in sorted(on_disk - set(registered))
        ]

    for path in sorted((package_dir / "mappings").glob("*.tsv")):
        for row in read_tsv(path):
            if row["datasetID"] != path.stem:
                findings.append(
                    Finding(
                        "dataset_is_not_file_stem",
                        SEVERITY_ERROR,
                        "mapping",
                        path.name,
                        f"{row['verbatimIdentification']!r} carries datasetID {row['datasetID']!r}",
                    )
                )
            named = taxa.get(row["taxonID"], {}).get("scientificName")
            if named is not None and row["concept"] != named:
                findings.append(
                    Finding(
                        "display_name_disagrees",
                        SEVERITY_ERROR,
                        "mapping",
                        path.name,
                        f"{row['verbatimIdentification']!r} names {row['concept']!r}, {row['taxonID']} is {named!r}",
                    )
                )
    return findings


def _check_canonical_names(taxa: dict) -> list:
    """Canonical names are lowercase. Raw labels are exempt; they are not in this table."""
    return [
        Finding("name_not_lowercase", SEVERITY_WARN, "taxon", taxon_id, f"{row['scientificName']!r} is not lowercase")
        for taxon_id, row in sorted(taxa.items())
        if row["scientificName"] != row["scientificName"].lower()
    ]


def check_coverage(package_dir: Path, registered) -> list:
    """Every registered source has at least one non-draft mapping. No hard-coded row counts.

    ``draft`` rows are skipped, so a partially mapped source is committable — design B could not
    commit one, and 150 blank rows produced 150 errors and a renderer crash.
    """
    covered = set()
    for path in sorted((Path(package_dir) / "mappings").glob("*.tsv")):
        if any(row["status"] != "draft" for row in read_tsv(path)):
            covered.add(path.stem)

    return [
        Finding("source_not_covered", SEVERITY_ERROR, "mapping", name, "registered, but no accepted mapping row")
        for name in sorted(set(registered) - covered)
    ]


# The gate
def validate(package_dir, registered=None) -> Report:
    """Run every check. Returns a :class:`Report`; raises only if the package cannot be read."""
    package_dir = Path(package_dir)
    descriptor = load_descriptor(package_dir)
    findings = check_descriptor(package_dir, descriptor) + check_rules(package_dir, descriptor, registered)
    if registered is not None:
        findings += check_coverage(package_dir, registered)
    return Report(findings=sorted(findings, key=lambda f: (f.check, f.resource, f.locator, f.detail)))


# The waiver table whose finding ids are THIS validator's. The other two tables under waivers/
# are adjudications of checks that have not moved here yet, so consuming them would report all 79
# as stale on every run — a false alarm that teaches a reader to ignore the stale list.
STRUCTURAL_WAIVERS = "structural.tsv"


def read_waivers(package_dir: Path) -> dict:
    """``{finding_id: row}`` for the adjudications of this validator's own findings.

    ``waivers/horizontal.tsv`` (the 20 KI-31 entries) and ``waivers/authority.tsv`` (the 59
    external-authority entries) are deliberately NOT read here. They are the existing adjudications
    ported onto stable concept ids, waiting for the horizontal check to move onto the loader in
    step 5 and the authority check in step 7. Their finding ids belong to those tools, not to this
    one, so treating them as waivers here would make every one of them permanently "stale".
    """
    path = Path(package_dir) / "waivers" / STRUCTURAL_WAIVERS
    if not path.exists():
        return {}
    waivers = {}
    for row in read_tsv(path):
        row["__file__"] = path.name
        waivers[row["finding_id"]] = row
    return waivers


def apply_waivers(report: Report, waivers: dict) -> Report:
    """Split the findings by adjudication, and name the waivers that match nothing.

    A stale waiver is reported rather than ignored: an adjudication that no longer describes
    anything is a claim about the data that has quietly stopped being true.
    """
    unwaived, waived = [], []
    for finding in report.findings:
        (waived if finding.finding_id in waivers else unwaived).append(finding)

    matched = {finding.finding_id for finding in waived}
    stale = sorted(set(waivers) - matched)
    return Report(findings=unwaived, waived=waived, stale_waivers=stale)


def main(argv=None) -> int:
    """CLI: validate the bundled package (or one named with --package) and print a report."""
    import argparse

    from planktonzilla.planktonzilla_dataset import constants
    from planktonzilla.planktonzilla_dataset.taxonomy import loader

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, default=loader.PACKAGE_DIR)
    parser.add_argument("--no-waivers", action="store_true", help="Report every finding, adjudicated or not.")
    args = parser.parse_args(argv)

    report = validate(args.package, registered=sorted(constants.DATASET_IMPORT_CONFIGS))
    if not args.no_waivers:
        report = apply_waivers(report, read_waivers(args.package))

    for finding in report.findings:
        print(finding.describe())
    for finding_id in report.stale_waivers:
        print(f"[STALE] waiver {finding_id} matches no current finding — delete it")
    print(json.dumps(report.summary()))
    return 1 if report.errors or report.stale_waivers else 0


if __name__ == "__main__":
    raise SystemExit(main())
