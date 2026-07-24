"""
(c) Inria

Network-free tests for the pz_build_sankey generator (planktonzilla/planktonzilla_dataset/build_sankey.py).

Exercise the pure tree-building + assembly logic with a tiny hand-built fixture and an in-memory
per-class image-count map, with hand-computed expectations. No HuggingFace Hub, Google Fonts, or
inria.fr requests: fetch_fonts / fetch_logo are never called, and scan_dataset is driven only against
a mocked ``datasets`` streaming API (so its per-class Counter is verified without any network).
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import json
from collections import Counter
from pathlib import Path

from planktonzilla.planktonzilla_dataset import build_sankey as bs

RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def _row(dataset, proposed_label, root_class, **ranks):
    r = {"Dataset": dataset, "proposed_label": proposed_label, "root_class": root_class}
    for rk in RANKS:
        r[rk] = ranks.get(rk, "")
    return r


# fixture: a genus-level living label (terminates on an INTERNAL node), a species under the same
# genus, and one non-living leaf — spread over two source datasets.
ROWS = [
    _row(
        "whoi",
        "calanus finmarchicus",
        "living",
        Kingdom="animalia",
        Phylum="arthropoda",
        Genus="calanus",
        Species="finmarchicus",
    ),
    _row("whoi", "calanus", "living", Kingdom="animalia", Phylum="arthropoda", Genus="calanus"),
    _row("zoolake", "marine snow", "detritus"),
]
# per-(dataset, proposed_label_lower, root_class) image counts
PER_SOURCE = Counter(
    {
        ("whoi", "calanus finmarchicus", "living"): 100,
        ("whoi", "calanus", "living"): 50,
        ("zoolake", "marine snow", "detritus"): 30,
    }
)


def _find(node, name):
    if node["n"] == name:
        return node
    for k in node.get("k", []):
        hit = _find(k, name)
        if hit is not None:
            return hit
    return None


def _per_label(per_source):
    out = Counter()
    for (_ds, pl, rc), n in per_source.items():
        out[(pl, rc)] += n
    return out


def test_normal_tree_additive_counts_and_full_mapping():
    tree, mapped = bs.build_tree(ROWS, _per_label(PER_SOURCE))
    assert mapped == 180  # every image mapped onto a terminal node
    assert (tree["c"], tree["s"], tree["t"]) == (3, 180, 2)  # 3 rows, 180 images, 2 leaf taxa

    living = _find(tree, "living")
    assert (living["c"], living["s"], living["t"]) == (2, 150, 1)

    # genus-level label terminates on the (internal) genus node: c/s are cumulative (own + descendants)
    calanus = _find(tree, "calanus")
    assert (calanus["c"], calanus["s"], calanus["t"]) == (2, 150, 1)  # 50 own + 100 from the species
    fin = _find(tree, "finmarchicus")
    assert (fin["c"], fin["s"], fin["t"]) == (1, 100, 1) and "k" not in fin

    detritus = _find(tree, "detritus")
    assert (detritus["c"], detritus["s"], detritus["t"]) == (1, 30, 1)
    assert _find(tree, "marine snow")["s"] == 30


def test_source_tree_inserts_source_level():
    tree, mapped = bs.build_source_tree(ROWS, PER_SOURCE)
    assert mapped == 180
    assert (tree["c"], tree["s"], tree["t"]) == (3, 180, 2)
    # root's children are the SOURCE datasets, each with a root_class child (rank "source" / "root_class")
    src_names = {c["n"]: c for c in tree["k"]}
    assert set(src_names) == {"whoi", "zoolake"}
    assert all(c["r"] == "source" for c in tree["k"])
    assert (src_names["whoi"]["c"], src_names["whoi"]["s"], src_names["whoi"]["t"]) == (2, 150, 1)
    assert (src_names["zoolake"]["c"], src_names["zoolake"]["s"], src_names["zoolake"]["t"]) == (1, 30, 1)
    assert src_names["whoi"]["k"][0]["r"] == "root_class"


def test_no_samples_gives_zero_images_but_keeps_structure():
    tree, mapped = bs.build_tree(ROWS, Counter())
    assert mapped == 0
    assert tree["c"] == 3 and tree["s"] == 0 and tree["t"] == 2


def test_scan_dataset_uses_datasets_api_and_preserves_counter(monkeypatch):
    """scan_dataset drives the scan through datasets.load_dataset + split_dataset_by_node and returns
    the exact per-(dataset, proposed_label_lower, root_class) Counter — None -> "", whitespace
    stripped, labels lowercased — aggregated across shards. The datasets API is mocked; no network.
    """
    import datasets
    import datasets.distributed
    import pyarrow as pa

    # Two fake shards with edge cases: a None label, surrounding whitespace, and mixed-case labels.
    shards = [
        pa.table(
            {
                "dataset": ["whoi", "whoi", "whoi", " zoolake "],
                "proposed_label": ["Calanus Finmarchicus", "Calanus Finmarchicus", "Calanus Finmarchicus", None],
                "root_class": ["living", "living", "living", "detritus"],
            }
        ),
        pa.table(
            {
                "dataset": ["ecotaxa", "ecotaxa"],
                "proposed_label": ["marine snow", "MARINE SNOW"],
                "root_class": ["detritus", "detritus"],
            }
        ),
    ]

    class FakeBase:
        num_shards = len(shards)

        def select_columns(self, cols):
            assert cols == ["dataset", "proposed_label", "root_class"]  # only the 3 metadata columns
            return self

        def with_format(self, fmt):
            assert fmt == "arrow"
            return self

    class FakeNode:
        def __init__(self, rank):
            self.rank = rank

        def iter(self, batch_size):
            yield shards[self.rank]  # whole shard as one arrow batch

    def fake_load_dataset(repo_id, split, streaming):
        assert split == "train" and streaming is True
        return FakeBase()

    def fake_split(base, rank, world_size):
        assert world_size == FakeBase.num_shards  # world_size == num_shards -> one whole shard per rank
        return FakeNode(rank)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(datasets.distributed, "split_dataset_by_node", fake_split)

    got = bs.scan_dataset("fake/repo", workers=8)
    assert got == Counter(
        {
            ("whoi", "calanus finmarchicus", "living"): 3,  # mixed case collapses to one key
            ("zoolake", "", "detritus"): 1,  # " zoolake " stripped; None label -> ""
            ("ecotaxa", "marine snow", "detritus"): 2,
        }
    )
    assert sum(got.values()) == 6


def test_assemble_substitutes_all_placeholders():
    tmpl = (
        "<style>__FONTS__ .x{}</style>"
        '<img src="data:image/svg+xml;base64,__LOGO_B64__">'
        '<script id="treedata">__TREE__</script>'
        '<script id="treedata_src">__TREE_SRC__</script>'
    )
    data = {"meta": {"total_rows": 3}, "tree": {"n": "planktonzilla-17M", "r": "dataset", "c": 3}}
    data_src = {"meta": {"total_rows": 3}, "tree": {"n": "planktonzilla-17M", "r": "dataset", "c": 3, "k": []}}
    html = bs.assemble(tmpl, data, data_src, "@font-face{}", "QUJD")
    for placeholder in bs.PLACEHOLDERS:
        assert placeholder not in html
    assert '"total_rows":3' in html and '"r":"dataset"' in html
    assert "QUJD" in html and "@font-face{}" in html


def test_real_template_has_the_four_placeholders():
    tmpl = bs.TEMPLATE_PATH.read_text(encoding="utf-8")
    for placeholder in bs.PLACEHOLDERS:
        assert tmpl.count(placeholder) == 1, placeholder


def test_real_template_has_colorby_controls():
    """The shipped template exposes the Color-by (Branch/Dataset) toolbar + client-side dataset palette.

    Network-free token check (mirrors ``test_real_template_has_the_four_placeholders``): the build is
    output-preserving, so we only assert the template still carries the client-side coloring seams.
    """
    tmpl = bs.TEMPLATE_PATH.read_text(encoding="utf-8")
    for token in ('id="colorbtns"', "renderColor(", "renderLegend(", "datasetColor(", "COLORBY"):
        assert token in tmpl, token


def test_real_template_has_converging_source_seam():
    """The shipped template carries the converging dataset-Sankey (Source=On) client-side seam.

    Network-free token check: the merged converging view is derived entirely client-side from the two
    already-emitted trees (DATA + DATA_SRC), so build_sankey.py is unchanged and this is a pure token
    check (no HuggingFace / Google Fonts / inria.fr requests).
    """
    tmpl = bs.TEMPLATE_PATH.read_text(encoding="utf-8")
    for token in ("sourceInflows(", "layoutMerged(", "MERGED", "enterMerged("):
        assert token in tmpl, token
    assert "MERGED ? layoutMerged()" in tmpl, "MERGED ? layoutMerged()"


def test_save_sample_counts_roundtrips(tmp_path):
    """save_sample_counts writes the exact JSON shape load_sample_counts reads back: round-trip equality."""
    c = Counter(
        {
            ("EcoTaxa", "copepoda", "living"): 5,
            ("WHOI", "detritus", "detritus"): 9,
            ("ZooLake", "copepoda", "living"): 2,
        }
    )
    out = tmp_path / "counts.json"
    bs.save_sample_counts(c, out)

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(doc["counts"], list)
    for row in doc["counts"]:
        assert set(row) == {"dataset", "proposed_label", "root_class", "n"}

    assert bs.load_sample_counts(out) == c


def test_parser_accepts_save_samples():
    assert bs._build_parser().parse_args(["--save-samples", "x.json"]).save_samples == Path("x.json")


def test_main_save_samples_then_reload(tmp_path, monkeypatch):
    """main() persists the loaded counts to --save-samples and returns 0, fully offline.

    ``--samples-json`` means scan_dataset is never called; fetch_fonts / fetch_logo are stubbed so the
    run makes no network requests (mirroring this module's network-free contract).
    """
    monkeypatch.setattr(bs, "fetch_fonts", lambda: "")
    monkeypatch.setattr(bs, "fetch_logo", lambda url: "")
    in_json = tmp_path / "in.json"
    out_json = tmp_path / "out.json"
    bs.save_sample_counts(Counter({("whoi", "calanus", "living"): 7}), in_json)
    rc = bs.main(
        [
            "--csv",
            str(bs.DEFAULT_TAXONOMY_CSV_FILENAME),
            "--samples-json",
            str(in_json),
            "--save-samples",
            str(out_json),
            "--out",
            str(tmp_path / "o.html"),
        ]
    )
    assert rc == 0
    assert bs.load_sample_counts(out_json) == bs.load_sample_counts(in_json)
