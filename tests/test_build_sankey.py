"""
(c) Inria

Network-free tests for the pz_build_sankey generator (planktonzilla/explorer/build_sankey.py).

Exercise the pure tree-building + assembly logic with a tiny hand-built fixture and an in-memory
per-class image-count map, with hand-computed expectations. No HuggingFace Hub, Google Fonts, or
inria.fr requests — the network paths (scan_dataset / fetch_fonts / fetch_logo) are never called.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

from collections import Counter

from planktonzilla.explorer import build_sankey as bs

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
