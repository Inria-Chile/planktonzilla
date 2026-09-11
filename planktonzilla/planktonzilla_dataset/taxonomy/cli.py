"""
(c) Inria

``pz_taxonomy`` — the curator's front end to the taxonomy package.

One command per act a curator actually performs, each of which prints what it WOULD do and stops.
Nothing here writes without ``--apply``, which is the opposite of the tools it replaces: the three
builders write on every run, one of them destroying 644 rows when re-run with no arguments at all.

    pz_taxonomy check                        validate the package and the descriptor
    pz_taxonomy show <dataset>               what one source maps, and to which concepts
    pz_taxonomy fmt [--apply]                canonical form, after a union merge or a spreadsheet
    pz_taxonomy rename <id> <name> [--apply] rename a concept, keeping its id
    pz_taxonomy retire <id> <into> --reason  retire a concept into another, with a tombstone
    pz_taxonomy clear-id <id> <authority> --reason
                                             the only way to blank an identifier
    pz_taxonomy render [--out FILE]          the 19-column legacy CSV
    pz_taxonomy diff [--against FILE]        which PUBLISHED cells differ from a rendered CSV

``upsert-source`` is deliberately absent. A source's records come from its builder — the rule
tables, label grammars and donor resolution that are the actual curation — and there is no way to
express 229 FREPJ records on a command line. Builders call :func:`write.upsert_source` directly;
this CLI is for the acts that ARE one line.
"""

import argparse
import sys
from pathlib import Path

from planktonzilla.planktonzilla_dataset.taxonomy import loader, validate, write
from planktonzilla.planktonzilla_dataset.taxonomy.model import TaxonomyError, read_tsv
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


def cmd_fmt(args) -> int:
    """Canonical form. Run it after a ``merge=union`` resolution or a spreadsheet round trip."""
    return _report(write.fmt(args.package, apply=args.apply), args.apply)


def cmd_rename(args) -> int:
    return _report(write.rename(args.package, args.taxon_id, args.name, apply=args.apply), args.apply)


def cmd_retire(args) -> int:
    return _report(write.merge(args.package, args.taxon_id, args.into, args.reason, apply=args.apply), args.apply)


def cmd_clear_id(args) -> int:
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
    """
    against = args.against if args.against is not None else Path(loader.default_source())
    changes = write.diff_published(args.package, against.read_bytes())
    print(changes.describe(limit=args.limit))
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
