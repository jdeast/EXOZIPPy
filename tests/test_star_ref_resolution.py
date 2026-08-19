"""A star reference may be a name everywhere it says it may (review 3.5.1).

`rvinstrument` and `band` both documented `star_ndx` as "Index or name" in
their `config_schema()` -- which the GUI surfaces -- while every consumer
called `int()` on it, so a name crashed with a raw ValueError.  The
translation now goes through one shared helper,
`components.component.resolve_star_ref`, which `relations._resolve_star`
and the SED's `photType` reader also delegate to.
"""

import numpy as np
import pytest

from exozippy.components.band.band import Band
from exozippy.components.component import resolve_star_ref
from exozippy.components.rvinstrument.rvinstrument import RVInstrument


class _CM:
    """ConfigManager stub carrying only the raw system config."""

    user_params = {}

    def __init__(self, star_names=("A", "B", "C")):
        self.system_config = {"star": [{"name": n} for n in star_names]}

    def add_hint(self, *a, **k):
        pass

    def add_scale_hint(self, *a, **k):
        pass

    def seed_start_value(self, path, seed=0):
        return None


# --------------------------------------------------------------------------
# The shared translator
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ref, expected",
    [
        (1, 1),
        ("B", 1),
        ("star.B", 1),
        ("2", 2),
        ("star.2", 2),
        (np.int64(0), 0),
    ],
)
def test_every_advertised_spelling_resolves(ref, expected):
    """
    Given the spellings a star reference is documented to accept,
    When resolve_star_ref translates one,
    Then it is the index of that star.
    """
    assert resolve_star_ref(ref, ["A", "B", "C"], "where") == expected


def test_unknown_name_names_the_location_and_the_stars():
    """
    Given a name that is not a star,
    When resolve_star_ref translates it,
    Then the error names the config location, the bad value and the stars
    that do exist -- not "invalid literal for int()".
    """
    with pytest.raises(ValueError) as exc:
        resolve_star_ref("Q", ["A", "B"], "[band] band 'I' star_ndx")

    msg = str(exc.value)
    assert "band 'I' star_ndx" in msg
    assert "'Q'" in msg
    assert "['A', 'B']" in msg


def test_out_of_range_index_is_refused():
    """
    Given an index past the last star,
    When resolve_star_ref translates it,
    Then it raises rather than handing back an index that will only fail
    much later, inside a tensor gather.
    """
    with pytest.raises(ValueError, match="out of range"):
        resolve_star_ref(5, ["A", "B"], "where")


def test_a_boolean_is_not_star_one():
    """
    Given `star_ndx: true` (a typo -- bool is an int in Python),
    When resolve_star_ref translates it,
    Then it raises instead of silently selecting star 1.
    """
    with pytest.raises(ValueError, match="invalid star reference"):
        resolve_star_ref(True, ["A", "B"], "where")


def test_no_star_names_known_still_accepts_indices():
    """
    Given a caller running before the star instances are known,
    When resolve_star_ref translates an index,
    Then it passes through -- the historical behaviour, not a failure.
    """
    assert resolve_star_ref(2, [], "where") == 2
    assert resolve_star_ref("3", None, "where") == 3


# --------------------------------------------------------------------------
# The two schemas that advertised it
# --------------------------------------------------------------------------
def test_rvinstrument_star_ndx_accepts_a_name(tmp_path):
    """
    Given an rvinstrument whose star_ndx is a star NAME,
    When the component is constructed,
    Then star_ndx is that star's index -- what its config_schema promises.
    """
    inst = RVInstrument(
        [{"name": "HIRES", "file": str(tmp_path / "x.rv"), "star_ndx": "B"}],
        _CM(),
    )

    assert inst.star_ndx == [1]


def test_band_star_ndx_accepts_a_name():
    """
    Given a band whose star_ndx is a star NAME,
    When the band loads,
    Then the declared LD star is that star's index.
    """
    band = Band([{"name": "I", "filter": "I", "star_ndx": "C"}], _CM())
    band.load_data(system=None)

    assert band.star_ndx_declared == [2]
    assert band.star_indices == [2]


def test_band_star_ndx_typo_names_the_band():
    """
    Given a band whose star_ndx names no star,
    When the band loads,
    Then the error names the band and the stars that exist.
    """
    with pytest.raises(ValueError) as exc:
        Band([{"name": "I", "filter": "I", "star_ndx": "Q"}], _CM()).load_data(
            system=None
        )

    msg = str(exc.value)
    assert "band 'I' star_ndx" in msg
    assert "'Q'" in msg
