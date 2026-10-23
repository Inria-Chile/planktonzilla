"""
(c) Inria

extract_taxon_ids.py
=====================
Wikidata harvest: given a taxon name, its Qcode, and from the Qcode the external identifiers other
authorities publish for it — WoRMS (P850 -> aphia_ID), NCBI Taxonomy (P685 -> NCBI_ID) and BOLD
Systems (P3606 -> BOLD_ID).

These functions are the network half of ``resolve_frepj_ids``, which is their only caller.

**The two-step CSV pipeline this module used to carry is retired** (step 6 of
``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``, open decision 12 of ``docs/TAXONOMY_REPRESENTATION.md``).
It read ``data/planktonzilla_taxonomy_v20.csv`` — a path that has not existed for some time — and
wrote ``taxonomy_and_wikidata.csv`` and ``taxonomy_wiki_and_ids.csv``, two intermediate tables
nothing consumed and whose disagreement about empty string versus null was half of KI-7. The
identifier table of the normalised taxonomy package is what they were reaching for:
``planktonzilla/planktonzilla_dataset/taxonomy/data/identifier.tsv``, one row per
``(concept, authority)`` with its justification and date, written through
``taxonomy.write.add_ids`` and validated by ``pz_taxonomy check``.

Requirements:
    pip install polars requests
"""

import time

import polars as pl
import requests

from planktonzilla.planktonzilla_dataset.constants import TAXONOMY_RANKS
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# Taxonomy columns, from most general to most specific.
COLS = list(TAXONOMY_RANKS)

HEADERS = {"User-Agent": "plankton-script/1.0"}

# Keywords to check that a Wikidata result is biological.
BIOLOGICAL_KEYWORDS = ["taxon", "species", "genus", "family", "order", "phylum", "organism"]

# Wikidata properties we want to extract.
WIKIDATA_PROPERTIES = {
    "aphia_ID": "P850",  # WoRMS ID
    "NCBI_ID": "P685",  # NCBI Taxonomy ID
    "BOLD_ID": "P3606",  # BOLD Systems taxon ID
}

session = requests.Session()
_SEARCH_CACHE: dict[str, dict | None] = {}

#: How many times a single Wikidata search rides out an HTTP 429 before giving up. The
#: waits double (2s, 4s, … 64s), so the budget is ~2 minutes rather than the flat 2s the
#: recursive version used.
RATE_LIMIT_ATTEMPTS = 6


# ── Step 1: Wikidata Qcode per taxon ──────────────────────────────────────────


def search_wikidata_taxon(taxon: str) -> dict | None:
    """Search a taxon on Wikidata and return {qid, label, description, url} or None.

    Caches per taxon and retries on rate limit (HTTP 429).
    """
    if taxon in _SEARCH_CACHE:
        return _SEARCH_CACHE[taxon]

    params = {
        "action": "wbsearchentities",
        "search": taxon,
        "language": "en",
        "format": "json",
        "limit": 5,
    }

    # A bounded loop, not recursion (KI-3, first half). On 429 this used to sleep 2s and
    # call ITSELF, so a rate-limit that outlasted the interpreter's stack — about a
    # thousand frames, half an hour of hammering Wikidata at a flat 2s — ended in a
    # RecursionError unwound through a thousand `except Exception` handlers. The budget
    # here is deliberate rather than whatever the stack happened to allow, and the waits
    # grow, so it backs off where the recursion did not.
    #
    # KI-3's SECOND half — the loose `description` substring match below, which can
    # resolve the wrong entity — is untouched: tightening it changes which Qcodes resolve,
    # hence the published aphia_ID / NCBI_ID / BOLD_ID values. Still → HARDEN-01.
    data = None
    for attempt in range(RATE_LIMIT_ATTEMPTS):
        try:
            r = session.get(
                "https://www.wikidata.org/w/api.php",
                params=params,
                headers=HEADERS,
                timeout=10,
            )
            if r.status_code == 429:
                if attempt == RATE_LIMIT_ATTEMPTS - 1:
                    # Nothing follows the last wait but the give-up below, so waiting 64s to do it
                    # is 64s of nothing.
                    break
                wait = 2 ** (attempt + 1)
                logger.info(f"Wikidata rate-limited on {taxon!r}; waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                return None
            data = r.json()
            break
        except Exception as e:
            logger.warning(f"Wikidata search failed for taxon {taxon!r}, returning None: {e}")
            return None

    if data is None:
        # Rate-limited for the whole budget. NOT cached: this is a transport outcome, and
        # caching it would make a later retry in the same run impossible (KI-5's shape).
        logger.warning(f"Wikidata stayed rate-limited for {taxon!r} across {RATE_LIMIT_ATTEMPTS} attempts.")
        return None

    for result in data.get("search", []):
        description = result.get("description", "").lower()
        if any(k in description for k in BIOLOGICAL_KEYWORDS):
            qid = result["id"]
            output = {
                "qid": qid,
                "label": result.get("label", ""),
                "description": result.get("description", ""),
                "url": f"https://www.wikidata.org/wiki/{qid}",
            }
            _SEARCH_CACHE[taxon] = output
            return output

    _SEARCH_CACHE[taxon] = None
    return None


def fetch_wikidata_ids(taxa: pl.DataFrame) -> pl.DataFrame:
    """Add Wikidata URL/Matched Taxon/Matched Rank/wikidata_ID to each unique taxon.

    For each taxon it searches Wikidata (network, via ``search_wikidata_taxon``)
    from the deepest available rank (Species) up to Kingdom, stopping at the first
    biological match.

    Args:
        taxa: DataFrame with the taxonomy-rank columns (``COLS``).

    Returns:
        The input DataFrame with added ``Wikidata URL``, ``Matched Taxon``,
        ``Matched Rank``, ``Matched Label`` (the resolved entity's Wikidata label,
        used downstream to verify the hit is the intended taxon) and ``wikidata_ID``
        (the Qcode extracted from the URL) columns.
    """
    wikidata_urls, matched_taxon, matched_rank, matched_label = [], [], [], []

    total = taxa.height
    for idx, row in enumerate(taxa.iter_rows(named=True), start=1):
        logger.info(f"[wikidata] {idx}/{total}")

        # Taxa present, from Species down to Kingdom (deepest rank first).
        taxons = [(row[c], c) for c in COLS if row[c] != ""]

        found_url = found_taxon = found_rank = found_label = ""
        for taxon, rank in reversed(taxons):
            result = search_wikidata_taxon(taxon)
            if result:
                found_url, found_taxon, found_rank = result["url"], taxon, rank
                found_label = result.get("label", "")
                break

        wikidata_urls.append(found_url)
        matched_taxon.append(found_taxon)
        matched_rank.append(found_rank)
        matched_label.append(found_label)

    return taxa.with_columns(
        [
            pl.Series("Wikidata URL", wikidata_urls),
            pl.Series("Matched Taxon", matched_taxon),
            pl.Series("Matched Rank", matched_rank),
            pl.Series("Matched Label", matched_label),
        ]
    ).with_columns(pl.col("Wikidata URL").str.extract(r"(Q\d+)", 1).alias("wikidata_ID"))


# ── Step 2: WoRMS / NCBI / BOLD per Qcode ───────────────────────────────────────


def _extract_property(claims: dict, prop: str) -> str | None:
    """Extract a property value from the claims of a Wikidata entity.

    Returns the first claim's value for ``prop``, or ``None`` if the property is
    absent or the nested structure is unexpected.
    """
    if prop not in claims:
        return None
    try:
        return claims[prop][0]["mainsnak"]["datavalue"]["value"]
    except (KeyError, IndexError, TypeError) as e:
        # Narrowed from `except Exception` (KI-1). This is the one site in that entry where the
        # swallowed set is enumerable: the body is a pure subscript chain over containers that
        # came out of `json`, `prop in claims` is guaranteed one line above, and every
        # JSON-decodable shape for `claims[prop]` raises one of these three or succeeds. So the
        # narrowing propagates nothing new today — it documents the contract and turns a future
        # non-JSON caller into a visible failure instead of a silently nulled id. The other
        # sites in KI-1 stay broad: each wraps network or image decoding, whose failure set is
        # not enumerable, and each feeds a published column.
        logger.debug(f"Could not extract property {prop} from claims, returning None: {e}")
        return None


def fetch_external_ids(taxa_wiki: pl.DataFrame, batch_size: int = 50) -> pl.DataFrame:
    """Query Wikidata in batches and return a DF with wikidata_ID + aphia/NCBI/BOLD.

    Resolves the unique Qcodes via Wikidata ``wbgetentities`` in batches (network),
    with up to 5 retries and exponential backoff on HTTP 429. Batches that never
    succeed contribute rows with ``None`` for every external ID.

    Args:
        taxa_wiki: DataFrame with a ``wikidata_ID`` column (Qcodes).
        batch_size: Number of Qcodes requested per Wikidata call.

    Returns:
        ``taxa_wiki`` left-joined on ``wikidata_ID`` with the extracted
        ``aphia_ID``/``NCBI_ID``/``BOLD_ID`` columns (``WIKIDATA_PROPERTIES``).
    """
    qcodes = taxa_wiki.select("wikidata_ID").drop_nulls().unique().to_series().to_list()

    results = []
    abandoned: list[str] = []
    for i in range(0, len(qcodes), batch_size):
        batch = qcodes[i : i + batch_size]
        batch_idx = i // batch_size + 1
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={'|'.join(batch)}&format=json"
        logger.info(f"[ids] batch {batch_idx} ({len(batch)} Qcodes)")

        success = False
        attempts = 5
        for attempt in range(attempts):
            try:
                r = requests.get(url, headers=HEADERS, timeout=60)
                if r.status_code == 429:
                    wait = 2**attempt
                    logger.info(f"  429 -> waiting {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                entities = r.json().get("entities", {})
                for qcode, entity in entities.items():
                    claims = entity.get("claims", {})
                    results.append(
                        {
                            "wikidata_ID": qcode,
                            **{col: _extract_property(claims, prop) for col, prop in WIKIDATA_PROPERTIES.items()},
                        }
                    )
                success = True
                break
            except Exception as e:
                logger.warning(f"  error in batch {batch_idx}: {e}")
                time.sleep(2)

        if not success:
            # The nulls below are indistinguishable from "Wikidata holds no external id for this
            # Qcode" (KI-6). Adding a status COLUMN would change this function's output schema, so
            # the distinction is surfaced where it costs nothing instead: named at ERROR, and
            # collected so a caller can re-ask for exactly the abandoned ids.
            logger.error(
                f"  batch {batch_idx} abandoned after {attempts} attempts; {len(batch)} Qcode(s) will be "
                f"null and are NOT known to lack ids: {', '.join(batch)}"
            )
            abandoned.extend(batch)
            results.extend([{"wikidata_ID": qcode, **{col: None for col in WIKIDATA_PROPERTIES}} for qcode in batch])
        time.sleep(1)

    if abandoned:
        logger.error(
            f"{len(abandoned)} of {len(qcodes)} Qcode(s) were abandoned rather than resolved. Their "
            f"aphia/NCBI/BOLD columns are null for a TRANSPORT reason, not a factual one; re-run over "
            f"them before treating the result as complete."
        )

    df_ids = pl.DataFrame(results)
    return taxa_wiki.join(df_ids, on="wikidata_ID", how="left")
