"""
(c) Inria
"""

import pytest
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(autouse=True)
def _reset_global_hydra():
    """Clear the GlobalHydra singleton after every test.

    Several tests follow the pattern ``hydra.initialize(...)`` ... work ...
    ``GlobalHydra.instance().clear()``. If anything between those two lines raises,
    the trailing clear is skipped and GlobalHydra stays initialized — and the next
    Hydra-using test (e.g. the ``@hydra.main`` training tests) then fails with
    "GlobalHydra is already initialized", so one upstream failure cascades into many.
    Clearing in a teardown finalizer makes that leak impossible regardless of how a
    test exits. ``clear()`` is a no-op when nothing is initialized, so this is safe
    for tests that never touch Hydra.
    """
    yield
    GlobalHydra.instance().clear()


@pytest.fixture()
def hydra_conf_path():
    return "./../configs"
