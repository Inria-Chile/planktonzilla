"""
(c) Inria

The KI-number registry: one number, one entry, forever.

`KNOWN_ISSUES.md` and `RESOLVED_ISSUES.md` both state the rule — "numbers are never
reused or renumbered", because commits, code comments and test function names cite them.
The rule was written down but nothing enforced it, and it was broken three times in one
week: two concurrent branches each took the next number they saw, and each merge silently
gave one number two meanings until a human noticed. Renumbering afterwards is what the
rule forbids, so the cost lands on whoever merges second.

These tests make that collision fail for the author who creates it, in their own branch,
instead of at merge time:

  - every KI number appears exactly once across BOTH ledgers;
  - the numbers a test name cites still exist (so an entry cannot be silently renumbered
    out from under `test_ki29_*`);
  - the Index names the next free number, so the next author does not have to infer it.

Network-free: reads only the two committed Markdown files and this test directory.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import re
from collections import Counter
from pathlib import Path

_UTILS = root / "planktonzilla" / "planktonzilla_dataset" / "utils"
_KNOWN = _UTILS / "KNOWN_ISSUES.md"
_RESOLVED = _UTILS / "RESOLVED_ISSUES.md"
_TESTS = root / "tests"

# `## KI-29 — ...` at the start of a line is an entry heading; a mention in prose is not.
_HEADING = re.compile(r"^## KI-(\d+)\b", re.MULTILINE)
# `def test_ki29_resolved_...` — a test whose name pins a specific entry.
_TEST_CITATION = re.compile(r"def test_ki(\d+)_")


def _headings(path: Path) -> list[int]:
    return [int(number) for number in _HEADING.findall(path.read_text(encoding="utf-8"))]


def test_every_ki_number_is_used_exactly_once():
    """One number, one entry — across both ledgers, which together are the registry."""
    numbers = _headings(_KNOWN) + _headings(_RESOLVED)
    duplicated = {number: count for number, count in Counter(numbers).items() if count > 1}
    assert duplicated == {}, (
        f"KI number(s) used by more than one entry: {sorted(duplicated)}. Both ledgers say numbers are "
        "never reused or renumbered — give the new entry the next free number instead (the Index names it)."
    )


def test_numbers_cited_by_test_names_still_exist():
    """A test named for an entry pins that number: the entry may move file, never number."""
    cited = {
        int(number) for path in _TESTS.glob("test_*.py") for number in _TEST_CITATION.findall(path.read_text(encoding="utf-8"))
    }
    assert cited, "no test cites a KI number; this guard would be vacuous"
    known = set(_headings(_KNOWN) + _headings(_RESOLVED))
    missing = sorted(cited - known)
    assert missing == [], (
        f"test name(s) cite KI number(s) with no entry in either ledger: {missing}. An entry was renumbered "
        "or deleted — restore the number, or the citation now points at nothing."
    )


def test_the_index_names_the_next_free_number():
    """So the next author reads it rather than inferring it from a skim."""
    text = _KNOWN.read_text(encoding="utf-8")
    expected = max(_headings(_KNOWN) + _headings(_RESOLVED)) + 1
    match = re.search(r"next free KI number is \*\*KI-(\d+)\*\*", text)
    assert match, (
        "KNOWN_ISSUES.md's Index must state 'The next free KI number is **KI-N**' — that sentence is what "
        "stops two branches picking the same number."
    )
    assert int(match.group(1)) == expected, (
        f"the Index advertises KI-{match.group(1)} as next free, but the highest entry in either ledger makes it KI-{expected}."
    )
