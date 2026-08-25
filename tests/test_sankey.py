"""
(c) Inria

Network-free tests for the pz_sankey generator (planktonzilla/planktonzilla_dataset/sankey.py).

Everything here exercises the pure data core — domain derivation, the non-living-at-Domain
rule, ribbon construction and template assembly — against a hand-built fixture with
hand-computed expectations, plus a smoke pass over the real bundled taxonomy CSV. Nothing
touches the network: ``fetch_fonts`` / ``fetch_logo`` / ``scan_dataset`` are never called.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import csv
import json
from collections import Counter
from datetime import datetime

from planktonzilla.planktonzilla_dataset import sankey as sk

RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def _row(dataset, proposed_label, root_class, **ranks):
    r = {"Dataset": dataset, "proposed_label": proposed_label, "root_class": root_class}
    for rk in RANKS:
        r[rk] = ranks.get(rk, "")
    return r


# Fixture: a species-level living label, a genus-level living label that STOPS at Genus, a
# bacterium (a second domain), and two non-living labels — spread over two source datasets.
ROWS = [
    _row(
        "whoi",
        "calanus finmarchicus",
        "living",
        Kingdom="animalia",
        Phylum="arthropoda",
        Class="hexanauplia",
        Order="calanoida",
        Family="calanidae",
        Genus="calanus",
        Species="finmarchicus",
    ),
    _row(
        "whoi",
        "chaetoceros",
        "living",
        Kingdom="chromista",
        Phylum="ochrophyta",
        Class="bacillariophyceae",
        Order="chaetocerotanae",
        Family="chaetocerotaceae",
        Genus="chaetoceros",
    ),
    _row("zooscan", "trichodesmium", "living", Kingdom="bacteria", Phylum="cyanobacteria", Genus="trichodesmium"),
    _row("whoi", "detritus", "detritus"),
    _row("zooscan", "bubble", "artefact"),
]

COUNTS = Counter(
    {
        ("whoi", "calanus finmarchicus", "living"): 100,
        ("whoi", "chaetoceros", "living"): 50,
        ("zooscan", "trichodesmium", "living"): 7,
        ("whoi", "detritus", "detritus"): 1000,
        ("zooscan", "bubble", "artefact"): 3,
    }
)

IMG, MAP = sk.COLUMNS.__len__(), sk.COLUMNS.__len__() + 1


def _decode(payload, ribbon):
    """Turn one encoded ribbon back into its list of column values ('' where absent)."""
    return [("" if ribbon[c] == sk.ABSENT else payload["vocab"][c][ribbon[c]]) for c in range(len(sk.COLUMNS))]


def _paths(payload):
    return {tuple(_decode(payload, r)): (r[IMG], r[MAP]) for r in payload["ribbons"]}


# --------------------------------------------------------------------------- columns
def test_columns_match_the_specified_order():
    assert [k for k, _ in sk.COLUMNS] == [
        "dataset",
        "root_class",
        "domain",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    assert sk.COLUMNS[sk.DOMAIN_COL][0] == "domain"


# --------------------------------------------------------------------------- domain
def test_domain_is_derived_from_kingdom():
    assert sk.domain_of("animalia", "x") == "Eukaryota"
    assert sk.domain_of("chromista", "x") == "Eukaryota"
    assert sk.domain_of("protozoa", "x") == "Eukaryota"
    assert sk.domain_of("plantae", "x") == "Eukaryota"
    assert sk.domain_of("bacteria", "x") == "Bacteria"
    assert sk.domain_of("archaea", "x") == "Archaea"


def test_domain_falls_back_to_the_label_then_to_unclassified():
    # A living row with no Kingdom that names its domain outright is taken at its word …
    assert sk.domain_of("", "Eukaryota") == "Eukaryota"
    # … but one that does not is reported honestly rather than guessed into Eukaryota.
    assert sk.domain_of("", "zooplankton") == sk.UNCLASSIFIED
    assert sk.domain_of("", "") == sk.UNCLASSIFIED
    # An unfamiliar kingdom is surfaced, never silently dropped.
    assert sk.domain_of("fictitia", "x") == "Fictitia"


# --------------------------------------------------------------------------- the spec rule
def test_non_living_puts_proposed_label_at_the_domain_column_and_stops():
    payload = sk.build_ribbons(ROWS, COUNTS)
    paths = _paths(payload)

    detritus = [p for p in paths if p[1] == "Detritus"]
    assert len(detritus) == 1
    p = detritus[0]
    assert p[:3] == ("whoi", "Detritus", "Detritus")  # proposed_label sits AT Domain
    assert all(v == "" for v in p[3:])  # and nothing continues past it

    artefact = next(p for p in paths if p[1] == "Artefact")
    assert artefact[:3] == ("zooscan", "Artefact", "Bubble")
    assert all(v == "" for v in artefact[3:])


def test_living_rows_carry_a_real_domain_then_the_ranks():
    paths = _paths(sk.build_ribbons(ROWS, COUNTS))
    calanus = next(p for p in paths if p[-1].startswith("Calanus"))
    assert calanus == (
        "whoi",
        "Living",
        "Eukaryota",
        "Animalia",
        "Arthropoda",
        "Hexanauplia",
        "Calanoida",
        "Calanidae",
        "Calanus",
        "Calanus finmarchicus",
    )


def test_a_ribbon_ends_at_the_deepest_rank_the_taxonomy_fills():
    """No '(blank)' sink: a genus-level label simply stops after Genus."""
    paths = _paths(sk.build_ribbons(ROWS, COUNTS))
    chaeto = next(p for p in paths if p[8] == "Chaetoceros")
    assert chaeto[8] == "Chaetoceros"
    assert chaeto[9] == ""  # Species is absent, not blank-filled


def test_bacteria_reach_the_domain_column_as_their_own_domain():
    paths = _paths(sk.build_ribbons(ROWS, COUNTS))
    tricho = next(p for p in paths if p[3] == "Bacteria")
    assert tricho[2] == "Bacteria"  # Domain
    assert tricho[3] == "Bacteria"  # Kingdom (the CSV's five-kingdom scheme)
    assert tricho[4] == "Cyanobacteria"


# --------------------------------------------------------------------------- species identity
def test_species_is_rendered_as_a_binomial():
    ranks = ["Animalia", "Arthropoda", "Hexanauplia", "Calanoida", "Calanidae", "Calanus", "Finmarchicus"]
    assert sk._binomial(list(ranks))[-1] == "Calanus finmarchicus"


def test_a_shared_epithet_under_two_genera_stays_two_distinct_species():
    """A node is keyed by (column, value), so a bare epithet would merge unrelated species."""
    rows = [
        _row("a", "chaetoceros socialis", "living", Kingdom="chromista", Genus="chaetoceros", Species="socialis"),
        _row("a", "parvicorbicula socialis", "living", Kingdom="chromista", Genus="parvicorbicula", Species="socialis"),
    ]
    payload = sk.build_ribbons(rows, Counter())
    species = {p[9] for p in _paths(payload)}
    assert species == {"Chaetoceros socialis", "Parvicorbicula socialis"}


def test_a_species_without_a_genus_keeps_its_own_value():
    assert sk._binomial(["Chromista", "", "", "", "", "", "Mirabilis"])[-1] == "Mirabilis"


# --------------------------------------------------------------------------- weights
def test_ribbon_weights_conserve_both_the_image_counts_and_the_mappings():
    payload = sk.build_ribbons(ROWS, COUNTS)
    assert payload["meta"]["total_images"] == sum(COUNTS.values()) == 1160
    assert payload["meta"]["total_mappings"] == len(ROWS) == 5
    assert sum(r[IMG] for r in payload["ribbons"]) == 1160
    assert sum(r[MAP] for r in payload["ribbons"]) == 5


def test_rows_sharing_a_path_are_merged_into_one_ribbon():
    """Two raw labels routed to the same class collapse to a single ribbon of weight 2."""
    rows = [
        _row("whoi", "detritus", "detritus"),
        _row("whoi", "detritus", "detritus"),
    ]
    payload = sk.build_ribbons(rows, Counter({("whoi", "detritus", "detritus"): 9}))
    assert len(payload["ribbons"]) == 1
    assert payload["ribbons"][0][IMG] == 9
    assert payload["ribbons"][0][MAP] == 2


def test_a_class_with_no_images_still_appears_with_a_mapping_weight():
    payload = sk.build_ribbons(ROWS, Counter())
    assert payload["meta"]["total_images"] == 0
    assert payload["meta"]["total_mappings"] == 5
    assert payload["meta"]["samples_available"] is False
    assert len(payload["ribbons"]) == 5


def test_counts_with_no_taxonomy_row_are_skipped_not_crashed():
    payload = sk.build_ribbons(ROWS, COUNTS + Counter({("whoi", "ghost label", "living"): 42}))
    assert payload["meta"]["total_images"] == 1160  # the orphan is not silently folded in
    assert not any("Ghost" in v for col in payload["vocab"] for v in col)


# --------------------------------------------------------------------------- payload shape
def test_every_ribbon_index_resolves_inside_its_column_vocabulary():
    payload = sk.build_ribbons(ROWS, COUNTS)
    for ribbon in payload["ribbons"]:
        assert len(ribbon) == len(sk.COLUMNS) + 2
        for col in range(len(sk.COLUMNS)):
            idx = ribbon[col]
            assert idx == sk.ABSENT or 0 <= idx < len(payload["vocab"][col])


def test_meta_reports_the_column_cardinalities():
    payload = sk.build_ribbons(ROWS, COUNTS)
    assert payload["meta"]["column_cardinality"] == [len(v) for v in payload["vocab"]]
    assert payload["meta"]["n_datasets"] == 2
    assert payload["meta"]["n_ribbons"] == len(payload["ribbons"])


# --------------------------------------------------------------------------- assembly
def test_assemble_substitutes_every_placeholder():
    html = sk.assemble("<style>__FONTS__</style><img src='__LOGO_B64__'><script>__PAYLOAD__</script>", {"a": 1}, "FF", "LL")
    assert not any(p in html for p in sk.PLACEHOLDERS)
    assert "FF" in html and "LL" in html and '{"a":1}' in html


def test_assemble_defangs_a_closing_script_tag_in_the_data():
    html = sk.assemble("__FONTS__|__LOGO_B64__|__PAYLOAD__", {"x": "</script><b>"}, "", "")
    assert "</script>" not in html
    assert "<\\/script>" in html


def test_the_shipped_template_carries_every_placeholder():
    text = sk.TEMPLATE_PATH.read_text(encoding="utf-8")
    for placeholder in sk.PLACEHOLDERS:
        assert placeholder in text, placeholder


def test_the_shipped_template_wires_all_three_exports():
    """SVG and PNG are images of the view; Mermaid is the same graph as `sankey-beta` text."""
    text = sk.TEMPLATE_PATH.read_text(encoding="utf-8")
    for element in ('id="export"', 'id="exportPng"', 'id="exportMermaid"'):
        assert element in text, element
    assert "sankey-beta" in text  # the Mermaid diagram keyword the export emits


# --------------------------------------------------------------------------- dataset naming
def test_assemble_prints_the_bare_dataset_name_and_the_full_repo_id():
    html = sk.assemble("<title>__DATASET_NAME__</title><p>__DATASET_REPO__</p>", {}, "", "", "org/plankton-9K")
    assert "<title>plankton-9K</title>" in html
    assert "<p>org/plankton-9K</p>" in html


def test_assemble_links_the_repo_id_to_the_dataset_on_the_hub():
    html = sk.assemble('<a href="__DATASET_URL__">__DATASET_REPO__</a>', {}, "", "", "org/plankton-9K")
    assert html == '<a href="https://huggingface.co/datasets/org/plankton-9K">org/plankton-9K</a>'


def test_dataset_url_passes_an_absolute_url_through_untouched():
    assert sk.dataset_url("https://example.org/data/plankton") == "https://example.org/data/plankton"
    assert sk.dataset_url("org/plankton-9K") == "https://huggingface.co/datasets/org/plankton-9K"


def test_assemble_accepts_a_dataset_name_with_no_org():
    html = sk.assemble("__DATASET_NAME__|__DATASET_REPO__", {}, "", "", "plankton-9K")
    assert html == "plankton-9K|plankton-9K"


def test_assemble_escapes_the_dataset_so_it_cannot_inject_markup():
    html = sk.assemble("<title>__DATASET_NAME__</title><p>__DATASET_REPO__</p>", {}, "", "", '<b onclick="x">')
    assert "<b onclick=" not in html
    assert html.count("&lt;b onclick=&quot;x&quot;&gt;") == 2


def test_assemble_defaults_to_the_published_planktonzilla_dataset():
    html = sk.assemble("__DATASET_REPO__", {}, "", "")
    assert html == sk.DEFAULT_PLANKTONZILLA_DATASET_REPO_ID


def test_a_dataset_name_inside_the_payload_is_never_treated_as_a_placeholder():
    # The payload lands last, so data that happens to spell a placeholder stays literal.
    html = sk.assemble("__PAYLOAD__ __DATASET_NAME__", {"d": "__DATASET_NAME__"}, "", "", "org/real")
    assert html == '{"d":"__DATASET_NAME__"} real'


# --------------------------------------------------------------------------- provenance
def test_provenance_stamps_the_build_time_in_utc_to_the_second():
    stamp = sk.provenance("org/plankton-9K", offline=True)["generated_at"]
    parsed = datetime.fromisoformat(stamp)
    assert stamp.endswith("Z") and "T" in stamp
    assert parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0
    assert parsed.microsecond == 0


def test_provenance_never_reaches_the_hub_when_the_version_is_pinned_or_offline(monkeypatch):
    def explode(repo_id):
        raise AssertionError("the Hub must not be consulted here")

    monkeypatch.setattr(sk, "fetch_dataset_metadata", explode)
    assert sk.provenance("org/plankton-9K", version="v1.2")["dataset_version"] == "v1.2"
    assert sk.provenance("org/plankton-9K", offline=True)["dataset_version"] == ""


def test_provenance_carries_the_hub_version_revision_and_modified_date(monkeypatch):
    monkeypatch.setattr(
        sk,
        "fetch_dataset_metadata",
        lambda repo_id: {"version": "b204961", "revision": "b204961" + "f" * 33, "modified": "2026-07-27"},
    )
    meta = sk.provenance("org/plankton-9K")
    assert meta["dataset_version"] == "b204961"
    assert meta["dataset_revision"].startswith("b204961")
    assert meta["dataset_modified"] == "2026-07-27"


def test_provenance_degrades_to_blanks_when_the_hub_is_unreachable(monkeypatch):
    monkeypatch.setattr(sk, "fetch_dataset_metadata", lambda repo_id: {})
    meta = sk.provenance("org/plankton-9K")
    assert meta["dataset_version"] == meta["dataset_revision"] == meta["dataset_modified"] == ""
    assert meta["generated_at"]  # the build stamp is local, so it survives a dead Hub


def _args(**kw):
    return sk._build_parser().parse_args([f"--{k.replace('_', '-')}={v}" for k, v in kw.items()])


def test_resolve_dataset_name_prefers_the_explicit_name_then_the_scanned_repo():
    assert sk.resolve_dataset_name(_args(dataset_name="org/explicit", dataset_repo="org/scanned")) == "org/explicit"
    assert sk.resolve_dataset_name(_args(dataset_repo="org/scanned")) == "org/scanned"
    assert sk.resolve_dataset_name(_args()) == sk.DEFAULT_PLANKTONZILLA_DATASET_REPO_ID


# --------------------------------------------------------------------------- real CSV
def _real_rows():
    with open(sk.DEFAULT_TAXONOMY_CSV_FILENAME, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_bundled_taxonomy_builds_a_consistent_payload():
    payload = sk.build_ribbons(_real_rows(), Counter())
    paths = [_decode(payload, r) for r in payload["ribbons"]]
    assert len(paths) > 500

    for path in paths:
        # The spec's rule holds for every non-living row in the real CSV.
        if path[1].lower() != "living":
            assert path[sk.DOMAIN_COL], path
            assert all(v == "" for v in path[sk.DOMAIN_COL + 1 :]), path
        # A ribbon never resumes after it stops — a hole would silently skip a rank.
        values = path[sk.DOMAIN_COL :]
        first_gap = next((i for i, v in enumerate(values) if not v), len(values))
        assert all(v == "" for v in values[first_gap:]), path


def test_the_domain_column_mixes_real_domains_and_non_living_labels():
    payload = sk.build_ribbons(_real_rows(), Counter())
    domains = set(payload["vocab"][sk.DOMAIN_COL])
    assert {"Eukaryota", "Bacteria"} <= domains
    assert {"Detritus", "Fiber"} <= domains  # non-living proposed_labels share the column


def test_the_real_payload_is_json_serialisable_and_round_trips():
    payload = sk.build_ribbons(_real_rows(), Counter())
    assert json.loads(json.dumps(payload))["meta"]["n_ribbons"] == payload["meta"]["n_ribbons"]
