"""
Filter profile pickles must load on every supported Python (>=3.12).

Python 3.13 pickles a pathlib.Path under the module name ``pathlib._local``,
which 3.12 cannot import. The shipped Roman WFI.F087/F146 profiles were
written that way and raised ModuleNotFoundError on 3.12.
"""

import io
import pickle
from pathlib import Path

from exozippy.filters.filter import Filter, _PortableUnpickler


def test_a_pickle_naming_pathlib_local_loads_on_this_python():
    """
    Given a pickle whose Path is recorded under ``pathlib._local`` (what
    Python 3.13+ writes), built by hand so the test means the same on
    every interpreter,
    When it is read with the Filter unpickler,
    Then it loads and the value is a Path.
    """
    # ARRANGE: protocol 0 is text, so the module name can be swapped in
    raw = pickle.dumps({"filterDirectory": Path("/x")}, protocol=0)
    raw = raw.replace(b"cpathlib\n", b"cpathlib._local\n")
    assert b"pathlib._local" in raw

    # ACT
    state = _PortableUnpickler(io.BytesIO(raw)).load()

    # ASSERT
    assert state["filterDirectory"] == Path("/x")


def test_shipped_roman_profiles_load():
    """
    Given the Roman WFI.F087 and WFI.F146 profiles shipped in the package,
    When Filter reads them,
    Then they load and carry a transmission curve.
    """
    # ACT
    filters = [Filter("Roman/WFI.F087"), Filter("Roman/WFI.F146")]

    # ASSERT
    for f in filters:
        wave, trans = f.ProcessedFilterCurve
        assert len(wave) == len(trans) > 0


def test_new_pickles_store_the_directory_as_a_string():
    """
    Given a loaded Filter,
    When its pickle state is taken,
    Then filterDirectory is a str, so the pickle does not depend on which
    Python's pathlib wrote it.
    """
    # ACT
    state = Filter("Roman/WFI.F087").__getstate__()

    # ASSERT
    assert isinstance(state["filterDirectory"], str)
