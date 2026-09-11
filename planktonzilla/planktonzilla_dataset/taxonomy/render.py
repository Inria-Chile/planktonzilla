"""
(c) Inria

Every projection of the taxonomy model: the 19-column legacy CSV, the 16-column lookup, the
published-dataset columns, and the training label vocabularies.

These are the views the repository currently derives in four separate places — the loader in
``generate_planktonzilla``, the near-verbatim copy in ``frepj_validate``, the rank-string
expression in ``gen_planktonzilla_only_plankton``, and ``sankey``'s own re-prefixing. Deriving
them once, here, from one model is the point of the whole migration; step 3 deletes the copies.

The ordering rule that matters most: three different column orders exist and conflating any two
silently swaps id values between columns.

    the CSV header      wikidata, aphia, NCBI, BOLD, ecotaxa
    the 16-col lookup   wikidata, ecotaxa, aphia, NCBI, BOLD   (constants.ID_STR/NUM_COLS)
    the published set   wikidata, ecotaxa, aphia, NCBI, BOLD   (constants.CONSOLIDATED_COLUMNS)
"""

import csv
import io

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy.model import (
    LEGACY_HEADER,
    LEGACY_ID_COLUMNS,
    LEGACY_RANKS,
    LOOKUP_COLUMNS,
    MULTI_VALUED_COLUMNS,
    UNQUALIFIED,
    TaxonomyError,
)


def project7(store, taxon_id: str) -> dict:
    """Fill the seven legacy rank slots from the ancestors whose rank HAS a slot.

    A rank with no ``legacy_slot`` contributes no cell. That single rule is what lets the 12
    ``unranked`` concept nodes stay distinct without widening the frozen columns, and what
    renders a lineage-less bucket as seven blanks.

    The ``Species`` cell is the epithet, derived by removing the genus prefix — the legacy column
    holds the epithet alone, which is not a node identity (19 epithets occur under more than one
    genus), so the binomial is stored and the epithet projected.

    Raises:
        TaxonomyError: If a species node has no genus ancestor, leaving the epithet undefined.
    """
    legacy_slot = store.vocab["rank"]
    slots = dict.fromkeys(LEGACY_RANKS, "")
    genus = ""

    for node in store.ancestors(taxon_id):
        slot = legacy_slot.get(node.rank, "")
        if not slot:
            continue
        if slot == "Genus":
            genus = node.scientific_name
        if slot != "Species":
            slots[slot] = node.scientific_name
            continue
        if not genus:
            raise TaxonomyError(f"species {node.scientific_name!r} has no genus ancestor; the epithet is underivable")
        slots[slot] = node.scientific_name.removeprefix(f"{genus} ")

    return slots


def _legacy_id_cell(column: str, values) -> str:
    """Render an id list back into its legacy cell form, float suffix included.

    Raises:
        TaxonomyError: If a single-valued authority carries more than one id. A second
            ``exact`` id is schema-legal in any SSSOM-shaped table and would otherwise be
            dropped here silently, with the survivor decided by sort order rather than by the
            model — an over-claim the refuters found in every design.
    """
    if not values:
        return ""
    if column in MULTI_VALUED_COLUMNS:
        return ";".join(values)
    if len(values) > 1:
        raise TaxonomyError(f"{column} is single-valued but carries {len(values)} ids: {values}")
    return values[0] if column == "wikidata_ID" else f"{values[0]}.0"


def legacy_rows(store) -> list:
    """The 19-column wide rows, as dicts of strings, in the release's stored physical order.

    Computed from the model — the taxon tree, the identifier index, the override layer — and
    never by parsing rendered bytes, so a consumer reading ``rows()`` exercises the same
    derivation the golden gate checks rather than a round trip through the renderer.
    """
    if not store.is_package:
        raise TaxonomyError("this store carries no model to project; it was loaded from a wide CSV")

    blank_qualifier = store.vocab["qualifier"]
    rows = []

    for key in store.row_order:
        try:
            mapping = store.mappings[key]
        except KeyError:
            raise TaxonomyError(f"the row-order manifest names {key}, which no mapping file carries") from None

        override = store.overrides.get(key)
        ids = store.identifiers.get(mapping.taxon_id, {})
        row = {
            "Dataset": mapping.dataset,
            "Raw_Labels": mapping.verbatim,
            **project7(store, mapping.taxon_id),
            "proposed_label": mapping.concept,
            "plankton": "True" if mapping.plankton else "False",
            "living": "True" if mapping.root_class == "living" else "False",
            "root_class": mapping.root_class,
            "qualifier": blank_qualifier[mapping.qualifier],
        }
        for column in LEGACY_ID_COLUMNS:
            row[column] = override[column] if override is not None else _legacy_id_cell(column, ids.get(column, []))
        rows.append(row)

    return rows


def render_wide_csv(store) -> bytes:
    """The 19-column CSV, byte-exact: LF line endings, ``QUOTE_MINIMAL``, the frozen header order."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(LEGACY_HEADER)
    for row in store.rows():
        writer.writerow([row[column] for column in LEGACY_HEADER])
    return buffer.getvalue().encode("utf-8")


def _norm(value):
    """Empty or blank strings become None; everything else is left as is.

    Transcribed from ``generate_planktonzilla._norm``, whose behaviour the lookup must match.
    """
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _as_decimal_free_string(value):
    """``135336.0`` -> ``"135336"``; a non-numeric or blank cell -> ``None``.

    Mirrors polars ``cast(Int64, strict=False).cast(Utf8)``, which is how the legacy reader
    turns KI-12's float-serialised ids into text. A value that will not parse becomes null there
    too, so it does here.
    """
    value = _norm(value)
    if value is None:
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return None


def build_lookup(rows) -> dict:
    """``(Dataset, Raw_Labels) -> {16 columns}``, value- and type-identical to the legacy reader.

    Value types, measured on the committed table and pinned by the golden gate: ``plankton`` is
    always ``bool``; every other value is ``str`` or ``None``.

    A duplicate key keeps the LAST row and is not an error, matching the legacy reader — the
    guard for that belongs at the store, not in the mapper. A column the rows do not carry
    resolves to ``None`` for every row, which is how an 18-column test fixture keeps working.
    """
    lookup = {}
    for row in rows:
        values = {}
        for column in LOOKUP_COLUMNS:
            raw = row.get(column)
            if column in constants.ID_NUM_COLS:
                values[column] = _as_decimal_free_string(raw)
            elif column == "plankton":
                values[column] = raw if isinstance(raw, bool) else _norm(raw) == "True"
            else:
                values[column] = _norm(raw)
        lookup[(row["Dataset"], row["Raw_Labels"])] = values
    return lookup


def published_projection(rows) -> list:
    """The taxonomy columns of the published dataset, in ``constants.CONSOLIDATED_COLUMNS`` order.

    ``plankton`` is a ``bool`` and ``living`` is excluded, which is the published contract and
    differs from both the CSV header and the 16-column lookup.
    """
    columns = [column for column in constants.CONSOLIDATED_COLUMNS if column in LOOKUP_COLUMNS]
    projected = []
    for row in rows:
        values = build_lookup([row])[(row["Dataset"], row["Raw_Labels"])]
        projected.append({column: values[column] for column in columns})
    return projected


def label_vocabulary(rows, datasets=None) -> list:
    """The ``ClassLabel`` names of the training set: ``sorted(set())`` over the plankton rows.

    Transcribed from ``gen_planktonzilla_only_plankton.build_only_plankton``, including the
    ``Kingdom != ""`` filter that admits ``None`` and so produces the empty-string class at
    index 0. That is reproduced, not corrected: changing it would renumber all 849 names after
    it, and the class ids of four released models with them.

    Args:
        rows: Wide rows, as :meth:`TaxonomyStore.rows` yields them.
        datasets: Restrict to these sources; ``None`` means every source in ``rows``.
    """
    names = set()
    for row in rows:
        if datasets is not None and row["Dataset"] not in datasets:
            continue
        values = build_lookup([row])[(row["Dataset"], row["Raw_Labels"])]
        if values["plankton"] is not True or values["Kingdom"] == "":
            continue
        names.add(" ".join(values[rank] for rank in LEGACY_RANKS if values[rank] not in ("", None)))
    return sorted(names)


def legacy_qualifier(store, qualifier: str) -> str:
    """``unqualified`` -> the blank cell; every other term is itself."""
    return "" if qualifier == UNQUALIFIED else qualifier
