"""
(c) Inria

``pz_taxonomy`` — the curator's front end to the taxonomy package.

One command per act a curator actually performs, each of which prints what it WOULD do and stops.
Nothing here writes without ``--apply``, which is the opposite of the tools it replaces: the three
builders write on every run, one of them destroying 644 rows when re-run with no arguments at all.

    pz_taxonomy check                        validate the package and the descriptor
    pz_taxonomy show <dataset>               what one source maps, and to which concepts
    pz_taxonomy provenance [<dataset>]       why the rows say what they say, and how many cannot say
    pz_taxonomy fmt [--apply]                canonical form, after a union merge or a spreadsheet
    pz_taxonomy rename <id> <name> [--apply] rename a concept, keeping its id
    pz_taxonomy retire <id> <into> --reason  retire a concept into another, with a tombstone
    pz_taxonomy clear-id <id> <authority> --reason
                                             the only way to blank an identifier
    pz_taxonomy render [--out FILE]          the 19-column legacy CSV
    pz_taxonomy diff [--summary]             which PUBLISHED cells differ from a rendered CSV
    pz_taxonomy release <tag> [--apply]      freeze today's row order as a new release
    pz_taxonomy runbook [--out FILE]         the curation runbook, generated from the package

``upsert-source`` is deliberately absent. A source's records come from its builder — the rule
tables, label grammars and donor resolution that are the actual curation — and there is no way to
express 229 FREPJ records on a command line. Builders call :func:`write.upsert_source` directly;
this CLI is for the acts that ARE one line.
"""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from planktonzilla.planktonzilla_dataset.taxonomy import loader, validate, write
from planktonzilla.planktonzilla_dataset.taxonomy.model import (
    OVERRIDE_COLUMNS,
    ROW_ORDER_COLUMNS,
    TaxonomyError,
    read_tsv,
    write_tsv,
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def _report(changes, apply: bool) -> int:
    """Print a change set and say what to do next. Exit 0 either way — a dry run is not a failure."""
    print(changes.describe())
    if changes and not apply:
        print("\nNothing was written. Re-run with --apply to write it.")
    return 0


def cmd_check(args) -> int:
    """Validate the package: descriptor, schema, cross-file rules, coverage."""
    report = validate.apply_waivers(validate.validate(args.package), validate.read_waivers(args.package))
    errors = [finding for finding in report.findings if finding.severity == validate.SEVERITY_ERROR]
    warnings = [finding for finding in report.findings if finding.severity != validate.SEVERITY_ERROR]

    for finding in [*errors, *warnings]:
        print(f"{finding.severity:5s} {finding.finding_id}  {finding.check}  {finding.locator}  {finding.detail}")
    for waiver in report.stale_waivers:
        print(f"STALE {waiver}  a waiver that matches nothing; delete it")

    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s), {len(report.stale_waivers)} stale waiver(s).")
    return 1 if errors else 0


def cmd_show(args) -> int:
    """What one source maps. The question a curator asks before touching anything."""
    store = loader.load_taxonomy(args.package)
    rows = read_tsv(args.package / "mappings" / f"{args.dataset}.tsv")
    if not rows:
        print(f"{args.dataset} maps nothing.")
        return 0

    width = max(len(row["verbatimIdentification"]) for row in rows)
    for row in sorted(rows, key=lambda row: row["verbatimIdentification"]):
        lineage = " / ".join(node.scientific_name for node in store.ancestors(row["taxonID"]))
        draft = "  [draft]" if row["status"] == "draft" else ""
        print(f"{row['verbatimIdentification']:<{width}}  {row['taxonID']}  {lineage}{draft}")
    print(f"\n{len(rows)} class director{'y' if len(rows) == 1 else 'ies'} in {args.dataset}.")
    return 0


def cmd_provenance(args) -> int:
    """Why the rows say what they say — the ledger's own answer, for one source or all of them.

    The question used to be answerable only by reading two Markdown reports written by two builders
    in two styles, covering two sources. Since step 7 it is a column, so it is one command, and the
    rows nobody can answer for are counted rather than left as an implication.
    """
    from collections import Counter

    store = loader.load_taxonomy(args.package)
    datasets = [args.dataset] if args.dataset else store.datasets()

    total, recorded = 0, 0
    for dataset in datasets:
        path = args.package / "mappings" / f"{dataset}.tsv"
        if not path.exists():
            print(f"refused: {dataset} maps nothing", file=sys.stderr)
            return 2
        rows = read_tsv(path)
        methods = Counter(row["method"] for row in rows if row["method"])
        donors = Counter(row["donor"] for row in rows if row["donor"])
        total += len(rows)
        recorded += sum(methods.values())

        blank = len(rows) - sum(methods.values())
        summary = ", ".join(f"{method} {count}" for method, count in methods.most_common()) or "nothing recorded"
        print(f"{dataset:<24} {len(rows):>4} rows   {summary}{f', blank {blank}' if blank and methods else ''}")
        if args.dataset and donors:
            print("  donors: " + ", ".join(f"{donor} ({count})" for donor, count in donors.most_common(8)))
        if args.dataset:
            for row in rows:
                if row["remarks"]:
                    print(f"  {row['verbatimIdentification']}: {row['remarks']}")

    share = f"{recorded / total:.0%}" if total else "0%"
    print(f"\n{recorded} of {total} row(s) carry a method ({share}); {total - recorded} cannot be answered for.")
    return 0


def cmd_fmt(args) -> int:
    """Canonical form. Run it after a ``merge=union`` resolution or a spreadsheet round trip."""
    return _report(write.fmt(args.package, apply=args.apply), args.apply)


def cmd_rename(args) -> int:
    """Rename a concept, keeping its id. The id is the identity; the name never was."""
    return _report(write.rename(args.package, args.taxon_id, args.name, apply=args.apply), args.apply)


def cmd_retire(args) -> int:
    """Retire a concept into another, leaving a tombstone rather than a hole."""
    return _report(write.merge(args.package, args.taxon_id, args.into, args.reason, apply=args.apply), args.apply)


def cmd_clear_id(args) -> int:
    """Remove one authority's identifiers from one concept. The only way to blank an id."""
    return _report(write.clear_id(args.package, args.taxon_id, args.authority, args.reason, apply=args.apply), args.apply)


def cmd_render(args) -> int:
    """The 19-column legacy CSV, byte-exact."""
    rendered = loader.load_taxonomy(args.package).render_wide_csv()
    if args.out is None:
        sys.stdout.buffer.write(rendered)
        return 0
    args.out.write_bytes(rendered)
    print(f"wrote {len(rendered)} bytes, {rendered.count(10) - 1} rows, to «{args.out}»")
    return 0


def cmd_diff(args) -> int:
    """Which PUBLISHED cells differ from a rendered CSV — the question a reviewer actually has.

    Defaults to the committed wide CSV, so ``pz_taxonomy diff`` on a clean tree prints nothing and
    on a curated one prints exactly the published cells that would move.

    ``--summary`` prints the one line CI posts on a taxonomy PR: "N published cells changed on M
    rows". A 2,358-line CSV diff does not tell a reviewer whether a curation moved published data
    or only its provenance; this does, in a sentence.
    """
    against = args.against if args.against is not None else Path(loader.default_source())
    changes = write.diff_published(args.package, against.read_bytes())

    if args.summary:
        cells = [change for change in changes.changes if change.column != "*"]
        added = [change for change in changes.changes if change.column == "*" and change.after == "row"]
        removed = changes.removals()
        rows = {change.key for change in changes.changes}
        if not changes:
            print("No published cell changed.")
            return 0
        parts = [f"**{len(cells)} published cell(s) changed on {len(rows)} row(s)**"]
        if added:
            parts.append(f"{len(added)} row(s) added")
        if removed:
            parts.append(f"{len(removed)} row(s) REMOVED")
        print(", ".join(parts) + ".")
        for column, count in sorted(Counter(change.column for change in cells).items()):
            print(f"- `{column}`: {count}")
        return 0

    print(changes.describe(limit=args.limit))
    return 0


def runbook(package_dir) -> str:
    """The curation runbook, GENERATED from the package and this parser.

    Generated and not written, because a hand-written runbook rots silently: it names commands that
    were renamed, counts that moved, and vocabularies that grew, and nothing goes red. Everything
    below is read from the package or from the argument parser at the moment of writing, and a test
    asserts the committed file still equals this — so the runbook cannot drift from the tool it
    documents without the suite saying so.
    """
    from collections import Counter

    package_dir = Path(package_dir)
    store = loader.load_taxonomy(package_dir)
    mappings = [row for path in sorted((package_dir / "mappings").glob("*.tsv")) for row in read_tsv(path)]
    with_method = sum(1 for row in mappings if row["method"])
    methods = read_tsv(package_dir / "vocab" / "method.tsv")
    releases = sorted(path.name for path in (package_dir / "release").iterdir() if path.is_dir())
    vocabularies = sorted(
        path.name.removesuffix("_taxpath.tsv") for path in (package_dir / "vocab" / "labels").glob("*_taxpath.tsv")
    )
    predicates = Counter(row["predicate_id"] for row in read_tsv(package_dir / "identifier.tsv"))

    lines = [
        "# Curating the taxonomy",
        "",
        "**Generated by `pz_taxonomy runbook`. Do not edit — edit the code or the package and re-run.**",
        "",
        "Every number below is read from the committed package. A hand-written runbook that says "
        "«2,358 rows» goes on saying it after the table grows; this one cannot.",
        "",
        "## What is in the package",
        "",
        "| | |",
        "| --- | ---: |",
        f"| sources | {len(store.datasets())} |",
        f"| mapping rows | {len(mappings)} |",
        f"| concepts (`taxon.tsv`) | {len(store.taxa)} |",
        f"| identifiers | {sum(predicates.values())} |",
        f"| — of them exact matches | {predicates['skos:exactMatch']} |",
        f"| — of them broad matches | {predicates['skos:broadMatch']} |",
        f"| rows whose provenance is recorded | {with_method} |",
        f"| rows nobody can answer for | {len(mappings) - with_method} |",
        f"| frozen releases | {', '.join(releases)} |",
        f"| released label vocabularies | {', '.join(vocabularies)} |",
        "",
        "## The commands",
        "",
        "Every write is a dry run until `--apply`. A refusal is one line on stderr and exit 2.",
        "",
        "| command | what it does |",
        "| --- | --- |",
    ]
    parser = build_parser()
    commands = next(action for action in parser._actions if isinstance(action.choices, dict))
    for name, sub in commands.choices.items():
        # The handler's own first docstring line. Every command has one, and this table is why:
        # a command added without a sentence explaining itself shows up here as a blank.
        summary = (sub.description or "").strip().splitlines()
        lines.append(f"| `pz_taxonomy {name}` | {summary[0] if summary else '**undocumented**'} |")

    lines += [
        "",
        "## How a decision gets recorded",
        "",
        "`method` is a controlled vocabulary. A new way of deciding a row is a new term here, which is a "
        "reviewable one-line diff — not free text nobody can group by.",
        "",
        "| method | emitted by | means |",
        "| --- | --- | --- |",
    ]
    lines += [f"| `{row['method']}` | {row['emitted_by']} | {row['definition']} |" for row in methods]

    lines += [
        "",
        "A blank `method` is not a term. It means nobody can say why that row claims what it claims, and "
        "the count above is pinned by a test so it can only go down.",
        "",
        "## The three things that are frozen",
        "",
        "1. **A release's row order.** `release/<tag>/legacy_row_order.tsv`, pinned by its own sha. Never "
        "edited — `pz_taxonomy release <tag>` cuts a new one. Rows mapped after a release render after its "
        "block, in collation order.",
        "2. **A released label vocabulary.** `vocab/labels/<tag>_taxpath.tsv`. The class ids a model was "
        "trained against; regenerating one renumbers them. New names are a new tag.",
        "3. **The published CSV.** `planktonzilla_taxonomy.csv` is rendered from the package and is still the "
        "source of record every consumer reads. `pz_taxonomy diff --summary` says what a curation moved in it.",
        "",
        "## Adding a source",
        "",
        "1. Curate it however the source demands — a rule table, a spreadsheet, an EcoTaxa export.",
        "2. Call `taxonomy.write.upsert_wide_rows(package, rows, provenance=...)` from your builder. It writes "
        "`mappings/<source>.tsv` and nothing else under `mappings/`; it cannot reach another source.",
        "3. `pz_taxonomy check` — zero errors, or fix what it names.",
        "4. `pz_taxonomy diff --summary` — confirm the published cells that moved are the ones you meant.",
        "5. Re-render the CSV (`pz_taxonomy render --out`) and commit both.",
        "",
        "If the source has an independent list of its class directories, add it to "
        "`tests/test_taxonomy_source_coverage.py`. Without one, a mistyped class name publishes sixteen nulls "
        "for every one of its images and nothing goes red.",
        "",
        "## Correcting something already published",
        "",
        "| you want to | do |",
        "| --- | --- |",
        "| rename a concept | `rename <id> <name> --apply` — the id is the identity, the name never was |",
        "| retire a concept into another | `retire <id> <into> --reason … --apply` — leaves a tombstone |",
        "| remove an identifier | `clear-id <id> <authority> --reason … --apply` — the only way |",
        "| correct an identifier | `clear-id`, then re-add. A record contradicting a held id is refused |",
        "| tidy a table after a merge | `pz_taxonomy fmt --apply` |",
        "",
    ]
    return "\n".join(lines) + "\n"


def cmd_runbook(args) -> int:
    """Write the generated curation runbook."""
    text = runbook(args.package)
    if args.out is None:
        print(text, end="")
        return 0
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {len(text.splitlines())} lines to «{args.out}»")
    return 0


def cmd_release(args) -> int:
    """Freeze the current physical order into a new release, and pin it.

    A release is the answer to "which rows did we publish, in which order?", and it is needed
    because the order WITHIN a label is not derivable — 56 runs covering 545 rows of the frozen
    prefix are not in any key's order, and no key tried recovers them.

    Cutting one never edits an existing release. ``v1.0``'s manifest and its pin stay exactly as
    they are, which is what lets the loader keep proving the frozen block has not drifted; the new
    tag records today's order, including the rows that have landed since.
    """
    release = args.package / "release" / args.tag
    if release.exists():
        print(f"refused: release {args.tag} already exists. A release is frozen; cut a new tag.", file=sys.stderr)
        return 2

    store = loader.load_taxonomy(args.package)
    order = [
        {"sequence": str(index), "datasetID": dataset, "verbatimIdentification": verbatim}
        for index, (dataset, verbatim) in enumerate(store.row_order)
    ]
    digest = hashlib.sha256(
        "\n".join(f"{row['datasetID']}\t{row['verbatimIdentification']}" for row in order).encode("utf-8")
    ).hexdigest()

    if not args.apply:
        print(f"would freeze {len(order)} row(s) as release {args.tag}, pinned to {digest[:12]}…")
        print("\nNothing was written. Re-run with --apply to write it.")
        return 0

    write_tsv(release / "legacy_row_order.tsv", ROW_ORDER_COLUMNS, order)
    # The overrides are per-release and start empty: a correction pinned against v1.0's bytes says
    # nothing about a release cut after it.
    write_tsv(release / "legacy_overrides.tsv", OVERRIDE_COLUMNS, [])
    (release / "sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"froze {len(order)} row(s) as release {args.tag}, pinned to {digest[:12]}…")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pz_taxonomy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, default=loader.PACKAGE_DIR, help="The package directory to act on.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, handler, help_text, *, writes=False):
        sub = commands.add_parser(name, help=help_text, description=handler.__doc__)
        sub.set_defaults(handler=handler)
        if writes:
            sub.add_argument("--apply", action="store_true", help="Write the change. Without it nothing is written.")
        return sub

    add("check", cmd_check, "Validate the package.")

    show = add("show", cmd_show, "What one source maps.")
    show.add_argument("dataset")

    provenance = add("provenance", cmd_provenance, "Why the rows say what they say.")
    provenance.add_argument("dataset", nargs="?", default=None, help="One source; omit for every source.")

    add("fmt", cmd_fmt, "Rewrite every table in canonical form.", writes=True)

    rename = add("rename", cmd_rename, "Rename a concept, keeping its id.", writes=True)
    rename.add_argument("taxon_id")
    rename.add_argument("name")

    retire = add("retire", cmd_retire, "Retire a concept into another, with a tombstone.", writes=True)
    retire.add_argument("taxon_id")
    retire.add_argument("into")
    retire.add_argument("--reason", required=True, help="Why. Recorded in merged.tsv; there is no default.")

    clear = add("clear-id", cmd_clear_id, "Remove one authority's identifiers from one concept.", writes=True)
    clear.add_argument("taxon_id")
    clear.add_argument("authority", help="A legacy column (aphia_ID) or a CURIE prefix (worms).")
    clear.add_argument("--reason", required=True, help="Why. There is no default: this is how ids get lost.")

    rendered = add("render", cmd_render, "Render the 19-column legacy CSV.")
    rendered.add_argument("--out", type=Path, default=None, help="Write here instead of stdout.")

    diff = add("diff", cmd_diff, "Which published cells differ from a rendered CSV.")
    diff.add_argument("--against", type=Path, default=None, help="Default: the committed wide CSV.")
    diff.add_argument("--limit", type=int, default=40)
    diff.add_argument("--summary", action="store_true", help="The one line CI posts: N cells on M rows.")

    book = add("runbook", cmd_runbook, "Write the generated curation runbook.")
    book.add_argument("--out", type=Path, default=None, help="Write here instead of stdout.")

    release = add("release", cmd_release, "Freeze the current row order as a new release.", writes=True)
    release.add_argument("tag", help="The new release tag, e.g. v1.1. An existing tag is never edited.")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except TaxonomyError as failure:
        # A refusal is the tool working. It gets one line, not a traceback the curator has to read
        # past to find the sentence that says what to do instead.
        print(f"refused: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
