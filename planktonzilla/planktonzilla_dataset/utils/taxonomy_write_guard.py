"""
(c) Inria

Preflight guard for the tools that rewrite ``planktonzilla_taxonomy.csv`` in place.

Three builders locate "their own" block by scanning the committed byte stream and then
re-serialise the file around it. That works only while each one's idea of where its block
ends agrees with reality, and one of them is already wrong: ``build_frepj_taxonomy.write_csv``
copies the prefix up to the first ``frepj`` row and writes its own block after it, so
everything appended LATER is discarded. On the committed table a no-op re-run turns 2358 rows
into 1714 — the 44 ``daplankton`` and 600 ``tara_pacific_*`` rows are destroyed, the process
exits 0, and nothing warns (``docs/CODE_REVIEW.md`` finding 1.1, reproduced).

The real fix is the write API of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md`` step 6, which
replaces byte-splicing with key-addressed rows. Until then this module makes the failure
impossible rather than silent, in two independent layers:

``assert_owns_every_change``
    The data-integrity layer, and the one that matters. Compares the rendered output against
    the file on disk and refuses the write if ANY line belonging to a source the caller does
    not own is dropped, added, reordered or edited. It cannot be bypassed — there is no flag
    for it, because there is no legitimate reason for one builder to move another's rows.

``assert_unlocked``
    The blast-radius layer. A builder that rewrites the committed table in place must be
    asked to, explicitly, on the command line. Protects against the accidental invocation —
    a stale shell command, a copied runbook line — not against a wrong render.

Both raise :class:`TaxonomyWriteRefusedError` and write nothing. Refusing is the correct outcome
here: these tools re-derive a block that is already committed, so a refused run loses no work.
"""

import csv

# Opting in is deliberately verbose. A flag a reader can mistake for routine is not a guard.
UNLOCK_FLAG = "--i-know-this-rewrites-the-csv"

_UNLOCK_DEST = "i_know_this_rewrites_the_csv"


class TaxonomyWriteRefusedError(RuntimeError):
    """Raised INSTEAD of writing, when a rewrite is unauthorised or would move foreign rows."""


def add_unlock_argument(parser) -> None:
    """Register the shared opt-in flag on a builder's argument parser."""
    parser.add_argument(
        UNLOCK_FLAG,
        dest=_UNLOCK_DEST,
        action="store_true",
        help=(
            "Actually rewrite the committed taxonomy CSV in place. Without this the builder "
            "reports what it would write and exits without touching the file."
        ),
    )


def is_unlocked(args) -> bool:
    """Read the opt-in flag off a parsed argparse namespace."""
    return bool(getattr(args, _UNLOCK_DEST, False))


def assert_unlocked(unlocked: bool, *, tool: str, path) -> None:
    """Refuse to rewrite the committed table unless the caller passed the opt-in flag.

    Args:
        unlocked: Whether the caller opted in.
        tool: The builder's name, for the message.
        path: The file that would have been rewritten.

    Raises:
        TaxonomyWriteRefusedError: If ``unlocked`` is false.
    """
    if not unlocked:
        raise TaxonomyWriteRefusedError(
            f"{tool} refused to rewrite «{path}» in place. This builder re-serialises the whole "
            f"file around its own block, which is why the rewrite is opt-in: pass {UNLOCK_FLAG} "
            f"to go ahead. Nothing was written."
        )


def _first_field(line: str) -> str:
    """The line's ``Dataset`` column, parsed as CSV rather than sliced.

    A literal ``frepj,`` inside some other quoted field must never read as a Dataset value,
    which is the same reasoning ``build_frepj_taxonomy.write_csv`` applies when it looks for
    its own block (WR-04).
    """
    fields = next(csv.reader([line]), [""])
    return fields[0] if fields else ""


def _foreign_lines(content: str, owner) -> list:
    """Every line of ``content`` whose ``Dataset`` is not one the caller owns.

    The header rides along in this set on purpose: its first field is ``Dataset``, which no
    builder owns, so a render that loses or rewrites the header is caught by the same check.
    """
    return [line for line in content.splitlines() if _first_field(line) not in owner]


def assert_owns_every_change(original: str, rendered: str, *, owner, tool: str) -> None:
    """Refuse the write unless every line outside ``owner`` survives unchanged and in order.

    Counting rows per source would miss an edit that keeps the count, so the comparison is on
    the lines themselves. Drops, additions, reorderings and in-place edits of another source's
    rows are all caught.

    Args:
        original: The file's current text.
        rendered: The text the builder is about to write.
        owner: The ``Dataset`` values this builder is allowed to rewrite.
        tool: The builder's name, for the message.

    Raises:
        TaxonomyWriteRefusedError: If any line outside ``owner`` would move.
    """
    owner = set(owner)
    before = _foreign_lines(original, owner)
    after = _foreign_lines(rendered, owner)
    if before == after:
        return

    kept = set(after)
    lost = [line for line in before if line not in kept]
    losses = {}
    for line in lost:
        dataset = _first_field(line)
        losses[dataset] = losses.get(dataset, 0) + 1

    if losses:
        detail = ", ".join(f"{dataset} ({count} row(s))" for dataset, count in sorted(losses.items()))
        summary = f"would DESTROY {len(lost)} row(s) belonging to {detail}"
    else:
        summary = f"would rewrite or reorder {len(before)} line(s) it does not own"

    raise TaxonomyWriteRefusedError(
        f"{tool} {summary}. It may only rewrite rows of {sorted(owner)}, and every other line "
        f"must survive byte-for-byte and in order. This is the failure mode of "
        f"docs/CODE_REVIEW.md finding 1.1: the builder re-serialises the file around its own "
        f"block and silently drops whatever was appended after it. Nothing was written."
    )
