"""
(c) Inria

The two external-authority harvesters — ``extract_cox`` (NCBI Entrez) and ``extract_taxon_ids``
(Wikidata) — had no tests at all, which is how four of the KI register's entries stayed open long
after they had stopped being hard: nothing would have gone red either way.

Everything here is network-free. The three properties are the ones the register names: a stalled
socket must become a failure rather than a hang, a CLI flag must reach the code path it names, and
a transport failure must be distinguishable from a factual "there is no such id".
"""

import socket

import pyrootutils
import pytest

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from planktonzilla.planktonzilla_dataset.utils import extract_cox as ec
from planktonzilla.planktonzilla_dataset.utils import extract_taxon_ids as eti


# --- KI-2: the NCBI path had no socket timeout ---------------------------------------
def test_the_entrez_timeout_is_imposed_and_then_restored():
    """Biopython retries but calls `urlopen` with no timeout and exposes no per-call hook.

    So the bound has to be imposed around the call. It is process-global while it holds, which is
    why the restore matters: this module is imported by tooling that makes its own connections.
    """
    before = socket.getdefaulttimeout()

    with ec._socket_timeout(5):
        assert socket.getdefaulttimeout() == 5

    assert socket.getdefaulttimeout() == before


def test_the_entrez_timeout_is_restored_even_when_the_call_raises():
    """A timeout left behind after a failed fetch would silently apply to everything after it."""
    before = socket.getdefaulttimeout()

    with pytest.raises(RuntimeError):
        with ec._socket_timeout(5):
            raise RuntimeError("the fetch failed")

    assert socket.getdefaulttimeout() == before


def test_both_entrez_calls_run_under_the_timeout():
    """Discriminating: asserts the timeout is ACTIVE at the moment Entrez is called, which a test
    that only checked the context manager in isolation would not catch if a call sat outside it."""
    seen = {}

    def _record(name):
        def _call(**kwargs):
            seen[name] = socket.getdefaulttimeout()
            raise OSError("no network in tests")

        return _call

    original_search, original_fetch = ec.Entrez.esearch, ec.Entrez.efetch
    original_sleep = ec.time.sleep
    try:
        ec.Entrez.esearch, ec.Entrez.efetch = _record("esearch"), _record("efetch")
        ec.time.sleep = lambda _seconds: None
        ec.search_nuccore("txid1[Organism:exp]", max_results=1)
        ec.fetch_sequences(["ABC123"], label="test")
    finally:
        ec.Entrez.esearch, ec.Entrez.efetch = original_search, original_fetch
        ec.time.sleep = original_sleep

    assert seen == {"esearch": ec.ENTREZ_TIMEOUT, "efetch": ec.ENTREZ_TIMEOUT}


# --- KI-4: --noexp never reached the batch path --------------------------------------
def test_process_csv_threads_expand_to_children_into_the_batch_path():
    """`--noexp` was accepted, logged nowhere, and did nothing on every batch run.

    `process_csv` hard-coded `expand_to_children=True` while `main` computed `not args.noexp` for
    the single-taxon path only.
    """
    import inspect

    assert "expand_to_children" in inspect.signature(ec.process_csv).parameters

    source = inspect.getsource(ec.process_csv)
    assert "expand_to_children=expand_to_children" in source, "process_csv still hard-codes the flag"
    assert "expand_to_children=True," not in source

    # And main hands it through, so the CLI flag reaches the gate.
    assert "expand_to_children=not args.noexp" in inspect.getsource(ec.main)


# --- KI-6: an abandoned batch looked exactly like "no such id" ------------------------
def test_an_abandoned_wikidata_batch_is_named_rather_than_silently_nulled(monkeypatch, caplog):
    """The nulls are unavoidable — a status column would change the output schema — but the
    DISTINCTION is not. A caller reading nulls has to be able to tell which ones are factual."""
    import polars as pl

    def _always_fails(*args, **kwargs):
        raise OSError("Wikidata is unreachable")

    monkeypatch.setattr(eti.requests, "get", _always_fails)
    monkeypatch.setattr(eti.time, "sleep", lambda _seconds: None)

    taxa = pl.DataFrame({"wikidata_ID": ["Q1", "Q2"]})
    with caplog.at_level("ERROR"):
        out = eti.fetch_external_ids(taxa, batch_size=2)

    assert out.height == 2
    assert all(out[column].is_null().all() for column in eti.WIKIDATA_PROPERTIES)
    assert "abandoned" in caplog.text
    assert "Q1" in caplog.text and "Q2" in caplog.text
    assert "NOT known to lack ids" in caplog.text


# --- KI-1: the one handler whose swallowed set is enumerable --------------------------
@pytest.mark.parametrize(
    "claims",
    [
        {"P850": []},
        {"P850": [{}]},
        {"P850": [{"mainsnak": {}}]},
        {"P850": [{"mainsnak": {"datavalue": {}}}]},
        {"P850": 1},
        {"P850": "a string"},
        {"P850": {"a": 1}},
        {"P850": None},
    ],
)
def test_a_malformed_claim_returns_none_rather_than_raising(claims):
    """`_extract_property`'s handler was narrowed from `except Exception` to three named types.

    The narrowing is only safe if it is EXHAUSTIVE — anything escaping it would be caught two
    frames up by `fetch_external_ids`' own bare handler, which burns all five retries and nulls
    the whole 50-Qcode batch. So every JSON-decodable shape is enumerated here rather than argued.
    """
    assert eti._extract_property(claims, "P850") is None


def test_a_well_formed_claim_still_returns_its_value():
    """The control: the narrowing must not have turned the success path into a swallow."""
    claims = {"P850": [{"mainsnak": {"datavalue": {"value": "115348"}}}]}

    assert eti._extract_property(claims, "P850") == "115348"
