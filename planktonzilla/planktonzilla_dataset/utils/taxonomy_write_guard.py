"""
(c) Inria

How a builder writes the master ``planktonzilla_taxonomy.csv``: from the package, never in place.

Until step 6 three builders located "their own" block by scanning the committed byte stream and
re-serialised the file around it. That works only while each one's idea of where its block ends
agrees with reality, and one of them was already wrong: ``build_frepj_taxonomy.write_csv`` copied
the prefix up to the first ``frepj`` row and wrote its own block after it, so everything appended
LATER was discarded. On the committed table a no-op re-run turned 2358 rows into 1714 — the 44
``daplankton`` and 600 ``tara_pacific_*`` rows destroyed, the process exiting 0, nothing warning
(``docs/CODE_REVIEW.md`` finding 1.1, reproduced).

Byte-splicing is gone. A builder now upserts its own sources into the normalised package
(:func:`taxonomy.write.upsert_wide_rows`, which writes ``mappings/<dataset>.tsv`` and nothing else
under ``mappings/``) and the master CSV is RE-RENDERED whole from the package by :func:`render_master`
below. A render is a total function of the package, so there is no byte range to get wrong and no
block boundary to misjudge.

With the splice gone the opt-in flag it needed — ``--i-know-this-rewrites-the-csv``, step 0.2 —
comes off too, as the plan said it would. What stays is the check that never needed a flag:

``assert_owns_every_change``
    Refuses the write if ANY line belonging to a source the caller does not own is dropped, added,
    reordered or edited. On the new path this should be unreachable, which is exactly why it runs:
    "this writer keeps foreign lines" was documented of the frepj writer too, and stayed true until
    a source landed after its block. An asserted invariant survives that; a documented one does not.

It raises :class:`TaxonomyWriteRefusedError` and writes nothing. Refusing is the correct outcome
here: these tools re-derive a block that is already committed, so a refused run loses no work.
"""

import csv
from pathlib import Path

from planktonzilla.planktonzilla_dataset.taxonomy import loader


class TaxonomyWriteRefusedError(RuntimeError):
    """Raised INSTEAD of writing, when a render would move rows the caller does not own."""


def _first_field(line: str) -> str:
    """The line's ``Dataset`` column, parsed as CSV rather than sliced.

    A literal ``frepj,`` inside some other quoted field must never read as a Dataset value.
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


def render_master(package_dir, csv_path, *, owner, tool: str) -> int:
    """Re-render the master CSV whole from the package, refusing if a foreign row would move.

    The replacement for every builder's in-place splice. The bytes come from the package, which is
    where the builder has just upserted its own sources, so a source it did not touch renders
    exactly as before — and :func:`assert_owns_every_change` proves that rather than assuming it.

    Args:
        package_dir: The normalised package to render.
        csv_path: The master CSV to overwrite.
        owner: The ``Dataset`` values this builder is allowed to change.
        tool: The builder's name, for the message.

    Returns:
        The number of rows written.

    Raises:
        TaxonomyWriteRefusedError: If any line outside ``owner`` would move. Nothing is written.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        # Not a missing convenience. The ownership check compares the render against what is on
        # disk, and there is nothing to compare against — so this would create a master CSV at a
        # path nobody has vouched for, unchecked. `pz_taxonomy render --out` is the way to write a
        # rendering somewhere new.
        raise TaxonomyWriteRefusedError(
            f"{tool} will not create «{csv_path}»: it re-renders an existing master CSV and checks "
            f"that no row it does not own moved, which needs the file it is replacing. Use "
            f"`pz_taxonomy render --out` to write a rendering to a new path. Nothing was written."
        )
    loader.cache_clear()
    rendered = loader.load_taxonomy(Path(package_dir)).render_wide_csv().decode("utf-8")
    original = csv_path.read_text(encoding="utf-8")

    assert_owns_every_change(original, rendered, owner=owner, tool=tool)
    csv_path.write_text(rendered, encoding="utf-8", newline="")
    return max(rendered.count("\n") - 1, 0)
