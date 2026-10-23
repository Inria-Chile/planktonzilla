"""
(c) Inria

The PlanktonSet-1 mirror walks NOAA's FTP listing and writes what it finds to disk, so
every directory and file name it uses as a local path component arrives from the SERVER.

``Path('/out') / '/etc/x'`` is ``/etc/x``, and ``..`` walks out of the mirror just as
freely — so a listing that carried either would write outside the destination. Unlikely
over NOAA's FTP, trivial to close, and closed at the parse rather than at the two places
that consume the names.
"""

import importlib.util

import pyrootutils
import pytest

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

_spec = importlib.util.spec_from_file_location("mirror_planktonset1", root / "scripts" / "mirror_planktonset1.py")
mirror = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mirror)


class _FakeFTP:
    """Replays one canned LIST response."""

    def __init__(self, lines):
        self.lines = lines

    def cwd(self, path):
        self.visited = path

    def retrlines(self, command, callback):
        for line in self.lines:
            callback(line)


def listing(name, *, directory=False):
    kind = "d" if directory else "-"
    return f"{kind}rw-r--r-- 1 ftp ftp 1024 Jan 01 00:00 {name}"


@pytest.mark.parametrize("name", ["a.jpg", "class_01", "Copepoda", "x y.jpg"])
def test_ordinary_names_are_kept(name):
    assert mirror._is_safe_name(name) is True


@pytest.mark.parametrize("name", ["../etc/passwd", "/etc/passwd", "..", ".", "", "a/b", "a\\b"])
def test_names_that_are_not_one_path_component_are_rejected(name):
    assert mirror._is_safe_name(name) is False


def test_a_traversing_file_never_reaches_the_manifest(capsys):
    ftp = _FakeFTP([listing("good.jpg"), listing("../../etc/passwd")])

    _, files = mirror._listdir(ftp, "/pub/whatever")

    assert list(files) == ["good.jpg"]
    assert "skipping unsafe name" in capsys.readouterr().out


def test_a_traversing_directory_never_becomes_a_class(capsys):
    ftp = _FakeFTP([listing("Copepoda", directory=True), listing("..", directory=True)])

    dirs, _ = mirror._listdir(ftp, "/pub/whatever")

    assert dirs == ["Copepoda"]


def test_the_sizes_of_the_kept_files_are_still_read():
    """The control: rejecting names must not disturb the byte counts the resume relies on."""
    ftp = _FakeFTP([listing("good.jpg")])

    _, files = mirror._listdir(ftp, "/pub/whatever")

    assert files == {"good.jpg": 1024}
