"""
(c) Inria

Network-free tests for the generic ``custom_metadata`` column (v1.2).

``RedefineDataset._flatten_metadata`` turns the consumed metadata keys into consolidated
columns and folds EVERY other key into one JSON object per row;
``make_planktonzilla.ensure_custom_metadata`` back-fills a base that predates the column
with the empty object. These pin the contract stated in ``constants``: always a JSON
object, never null, keys sorted, blank values dropped, ``"{}"`` when there is nothing.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import json

import datasets
import pytest
from datasets import Value

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp
from planktonzilla.planktonzilla_dataset import make_planktonzilla as mk

COL = constants.CUSTOM_METADATA_COL


def _write_taxonomy_csv(path):
    header = (
        "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
        "proposed_label,plankton,root_class,qualifier,"
        "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
    )
    row = "src,copepoda,Animalia,Arthropoda,,,,,,Copepoda,True,zoo,,Q3386609,274;1231,135336.0,6854.0,"
    path.write_text(header + "\n" + row + "\n")


@pytest.fixture
def flatten(tmp_path, monkeypatch):
    """Run the REAL base-class flatten over a list of metadata JSON strings."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(csv_path)
    redefiner = gp.NoMetadataRedefiner(csv_taxonomies_path=str(csv_path))
    monkeypatch.setattr(gp, "num_proc", 1)

    def _flatten(metadata_json):
        return redefiner._flatten_metadata(datasets.Dataset.from_dict({"metadata": metadata_json}))

    return _flatten


def test_contract_constants():
    """custom_metadata is the last consolidated column and its empty value is a JSON object."""
    assert constants.CONSOLIDATED_COLUMNS[-1] == COL
    assert constants.CONSOLIDATED_COLUMNS.count(COL) == 1
    assert json.loads(constants.EMPTY_CUSTOM_METADATA) == {}


def test_leftover_keys_land_in_custom_metadata_sorted_and_blank_free(flatten):
    """Consumed keys become columns; everything else becomes ONE sorted JSON object, blanks dropped."""
    metadata = json.dumps(
        {
            "site": "biwako",
            "Latitude": "35.25",
            "magnification": "40",
            "Timestamp": "2021-11-01",
            "empty": "",
            "none": None,
            "zeta": "z",
            "alpha": "a",
        }
    )
    out = flatten([metadata])

    assert out[COL][0] == '{"alpha": "a", "magnification": "40", "site": "biwako", "zeta": "z"}'
    assert out["Latitude"][0] == pytest.approx(35.25)
    assert out["timestamp"][0] == "2021-11-01"
    for col in ("site", "magnification", "alpha", "zeta", "metadata"):
        assert col not in out.column_names


def test_every_consumed_key_is_excluded(flatten):
    """A metadata object made only of consumed keys leaves the empty object behind."""
    metadata = json.dumps(
        {
            "ObjID": "1",
            "BinID": "b",
            "Depth": "1",
            "Depth_max": "2",
            "Depth_min": "1",
            "Latitude": "1",
            "Longitude": "2",
            "Humidity": "3",
            "Temperature": "4",
            "Timestamp": "2020-01-01",
        }
    )
    out = flatten([metadata])
    assert out[COL][0] == constants.EMPTY_CUSTOM_METADATA
    assert set(gp._CONSUMED_METADATA_KEYS) == set(json.loads(metadata))


def test_nothing_to_add_and_malformed_json_give_the_empty_object(flatten):
    """{} / an empty string / unparseable JSON all yield "{}" — never null."""
    out = flatten([json.dumps({}), "", "not json"])
    assert out[COL] == [constants.EMPTY_CUSTOM_METADATA] * 3
    assert out.features[COL] == Value("string")


def test_ensure_custom_metadata_fills_a_base_that_predates_it():
    """A base without the column gets "{}" on every row; a base with it is returned untouched."""
    base = datasets.Dataset.from_dict({"dataset": ["isiisnet", "lensless"]})
    out = mk.ensure_custom_metadata(base, where="test base")
    assert out[COL] == [constants.EMPTY_CUSTOM_METADATA] * 2
    assert out.features[COL] == Value("string")
    assert mk.ensure_custom_metadata(out, where="test base") is out
