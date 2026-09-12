"""
(c) Inria

``search_wikidata_taxon`` used to answer an HTTP 429 by sleeping two seconds and calling
ITSELF. A rate limit that outlasted the interpreter's stack — about a thousand frames,
half an hour of hammering Wikidata at a flat 2 s — ended in a ``RecursionError`` unwound
through a thousand ``except Exception`` handlers (KI-3, first half).

KI-3's second half, the loose ``description`` substring match, is deliberately NOT fixed
here: tightening it changes which Qcodes resolve, hence the published `aphia_ID` /
`NCBI_ID` / `BOLD_ID` values, so it stays gated on the golden diff. The test at the bottom
pins that it is still loose, so this file cannot be read as closing KI-3.
"""

import pyrootutils
import pytest

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from planktonzilla.planktonzilla_dataset.utils import extract_taxon_ids as eti


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _Session:
    """Answers with a scripted sequence of responses, recording every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0) if self.responses else _Response(429)


def _match(qid="Q12345", description="species of diatom"):
    return _Response(200, {"search": [{"id": qid, "label": "test taxon", "description": description}]})


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    eti._SEARCH_CACHE.clear()
    monkeypatch.setattr(eti.time, "sleep", lambda seconds: None)
    yield
    eti._SEARCH_CACHE.clear()


def test_a_permanent_rate_limit_gives_up_instead_of_exhausting_the_stack(monkeypatch, caplog):
    """The reproduction: an endpoint that only ever answers 429."""
    session = _Session([])
    monkeypatch.setattr(eti, "session", session)

    with caplog.at_level("WARNING"):
        assert eti.search_wikidata_taxon("nitzschia") is None

    assert session.calls == eti.RATE_LIMIT_ATTEMPTS, "the budget is what bounds it"
    assert "stayed rate-limited" in caplog.text


def test_a_rate_limit_that_clears_still_resolves(monkeypatch):
    """Backing off is only worth doing if the retry can still succeed."""
    session = _Session([_Response(429), _Response(429), _match()])
    monkeypatch.setattr(eti, "session", session)

    assert eti.search_wikidata_taxon("nitzschia")["qid"] == "Q12345"
    assert session.calls == 3


def test_the_waits_grow_rather_than_staying_flat(monkeypatch):
    """The recursive version slept a flat 2s however long the limit lasted."""
    waited = []
    monkeypatch.setattr(eti.time, "sleep", waited.append)
    monkeypatch.setattr(eti, "session", _Session([]))

    eti.search_wikidata_taxon("nitzschia")

    assert waited == [2, 4, 8, 16, 32, 64]


def test_a_rate_limited_answer_is_not_cached(monkeypatch):
    """KI-5's shape: caching a transport outcome makes a later retry impossible.

    (KI-5 itself is not reproducible against this code — no transport path writes to
    `_SEARCH_CACHE` — and this pins that the new give-up path did not introduce it.)
    """
    monkeypatch.setattr(eti, "session", _Session([]))
    eti.search_wikidata_taxon("nitzschia")

    assert "nitzschia" not in eti._SEARCH_CACHE

    monkeypatch.setattr(eti, "session", _Session([_match()]))
    assert eti.search_wikidata_taxon("nitzschia")["qid"] == "Q12345"


def test_a_genuine_no_match_is_cached(monkeypatch):
    """The control: a 200 that matches nothing IS a real answer, and is remembered."""
    session = _Session([_Response(200, {"search": [{"id": "Q1", "description": "german footballer"}]})])
    monkeypatch.setattr(eti, "session", session)

    assert eti.search_wikidata_taxon("mueller") is None
    assert eti._SEARCH_CACHE["mueller"] is None

    eti.search_wikidata_taxon("mueller")
    assert session.calls == 1, "the cached no-match must not be re-queried"


def test_a_non_429_error_status_still_returns_none_without_retrying(monkeypatch):
    session = _Session([_Response(503)])
    monkeypatch.setattr(eti, "session", session)

    assert eti.search_wikidata_taxon("nitzschia") is None
    assert session.calls == 1


def test_the_loose_keyword_match_is_still_loose(monkeypatch):
    """KI-3's SECOND half is untouched and stays gated — this file does not close KI-3.

    `order` inside `disorder` is the shape that resolves the wrong entity. Tightening it
    changes which Qcodes resolve, hence published ID values.
    """
    assert "order" in eti.BIOLOGICAL_KEYWORDS
    monkeypatch.setattr(eti, "session", _Session([_match(description="a mental disorder")]))

    assert eti.search_wikidata_taxon("nitzschia") is not None, "still matching on a substring"
