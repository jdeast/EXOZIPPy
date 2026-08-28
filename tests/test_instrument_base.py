"""Tests for the shared Instrument base class (components/instrument.py).

These cover the behavior-preserving extraction from notes/instrument_todo.txt:
the two noise parameterizations (additive jitter_variance vs multiplicative
err_scale), the optional detrend block-diagonal builder, the shared jitter
floor, the optional per-instrument plot styling, and the factory's skipping of
the abstract base.  Physics/likelihood behavior of the four children is covered
by their own integration suites; here we pin the shared scaffolding directly.
"""

import inspect

import astropy.units as u
import numpy as np
import pytest

from exozippy.chart import Trace
from exozippy.components.component import Component
from exozippy.components.factory import discover_components
from exozippy.components.instrument import Instrument


class _DummyInstrument(Instrument):
    """Minimal concrete Instrument for unit-testing the shared helpers."""

    @property
    def prefix(self):
        return "dummy"

    def register_parameters(self, system):  # pragma: no cover - not exercised
        pass

    def build_likelihood(self, model, system):  # pragma: no cover
        pass


class _DummyErrScale(_DummyInstrument):
    noise_model = "err_scale"


def _make(config):
    return _DummyInstrument(config, config_manager=None)


# ---------------------------------------------------------------------------
# Factory: the abstract base must not be a discoverable component
# ---------------------------------------------------------------------------
def test_instrument_base_is_abstract():
    """
    Given the Instrument base leaves Component's abstract methods unimplemented,
    When inspect.isabstract is queried,
    Then Instrument is abstract (so the factory will skip it).
    """
    assert inspect.isabstract(Instrument)


def test_factory_does_not_register_instrument_base():
    """
    Given the factory scans every module (Instrument is imported into each child),
    When discover_components runs,
    Then no 'instrument' key is registered and every value is concrete.
    """
    registry = discover_components()
    assert "instrument" not in registry
    for key, cls in registry.items():
        assert not inspect.isabstract(cls), f"{key} -> {cls} is abstract"
        assert issubclass(cls, Component) and cls is not Component


def test_four_children_are_instrument_subclasses_with_expected_noise_model():
    """
    Given the four data components,
    When their class attributes are inspected,
    Then all subclass Instrument and mulens uses err_scale while the rest use
    the additive jitter_variance model.
    """
    registry = discover_components()
    expected = {
        "rvinstrument": "jitter_variance",
        "transit": "jitter_variance",
        "astrometryinstrument": "jitter_variance",
        "mulensinstrument": "err_scale",
    }
    for key, noise in expected.items():
        cls = registry[key]
        assert issubclass(cls, Instrument)
        assert cls.noise_model == noise


# ---------------------------------------------------------------------------
# Shared jitter floor
# ---------------------------------------------------------------------------
def test_jitter_floor_matches_formula():
    """
    Given an error array,
    When _jitter_floor is applied,
    Then it returns -0.95 * min(err)**2.
    """
    err = np.array([0.3, 0.1, 0.25])
    assert Instrument._jitter_floor(err) == pytest.approx(-0.95 * 0.1**2)


def test_jitter_floor_applies_unit_factor_before_squaring():
    """
    Given a unit conversion factor (rv scales the error column to m/s first),
    When _jitter_floor is applied with that factor,
    Then the floor is computed on the converted minimum.
    """
    err = np.array([2.0, 0.5, 1.0])
    factor = 3.0
    assert Instrument._jitter_floor(err, factor=factor) == pytest.approx(
        -0.95 * (0.5 * factor) ** 2
    )


# ---------------------------------------------------------------------------
# Shared detrend block-diagonal builder
# ---------------------------------------------------------------------------
def _whiten(col):
    """The whitening _build_block_detrend applies (see review 6.5.2)."""
    col = np.asarray(col, dtype=float)
    return (col - np.mean(col)) / np.std(col)


def test_build_block_detrend_places_columns_on_the_diagonal():
    """
    Given two instruments with 3 and 2 observations and 1 and 2 detrend columns,
    When _build_block_detrend runs,
    Then the design matrix is block-diagonal so coefficients never mix across
    instruments, and the column counts are reported.
    """
    a = np.array([[1.0], [2.0], [3.0]])  # inst 0: 3 obs, 1 col
    b = np.array([[4.0, 5.0], [6.0, 7.0]])  # inst 1: 2 obs, 2 cols
    inst = _make([{"name": "A"}, {"name": "B"}])
    matrix, per_inst, total, _ = inst._build_block_detrend([a, b], 5)

    assert per_inst == [1, 2]
    assert total == 3
    assert matrix.shape == (5, 3)
    # inst 0 column only populated in its own rows/cols
    assert np.allclose(matrix[:3, 0], _whiten([1.0, 2.0, 3.0]))
    assert np.all(matrix[:3, 1:] == 0.0)  # no cross-instrument leakage
    assert np.all(matrix[3:, 0] == 0.0)
    assert np.allclose(matrix[3:, 1], _whiten(b[:, 0]))
    assert np.allclose(matrix[3:, 2], _whiten(b[:, 1]))


def test_build_block_detrend_handles_no_detrend_columns():
    """
    Given instruments with zero detrend columns,
    When _build_block_detrend runs,
    Then it returns a (n_obs, 0) matrix and total 0.
    """
    empties = [np.empty((3, 0)), np.empty((2, 0))]
    inst = _make([{"name": "A"}, {"name": "B"}])
    matrix, per_inst, total, scales = inst._build_block_detrend(empties, 5)
    assert total == 0
    assert per_inst == [0, 0]
    assert matrix.shape == (5, 0)
    assert scales.shape == (0,)


# ---------------------------------------------------------------------------
# The shared concatenation template (ConcatenatedData)
#
# rvinstrument, transit and mulensinstrument all build their concatenated
# arrays through this accumulator, so the per-file row-range/contiguity
# invariant that _build_block_detrend and mulensing's observer_pos both depend
# on is a contract of the base class, and is pinned here rather than being
# re-derived (or assumed) per child.
# ---------------------------------------------------------------------------
_BLOCKS_CONFIG = [
    {"name": "A", "file": "a.dat"},
    {"name": "B", "file": "b.dat"},
    {"name": "C", "file": "c.dat"},
]
_BLOCK_ROWS = (4, 3, 5)


def _filled_blocks(inst, sides=False):
    """Feed inst's accumulator three files of 4/3/5 rows and return it."""
    blocks = inst._concat_blocks()
    start = 0
    for i, n in enumerate(_BLOCK_ROWS):
        rows = np.arange(start, start + n, dtype=float)
        extra = (
            {"observer_pos": np.column_stack([rows, rows + 0.5, rows + 0.25])}
            if sides
            else {}
        )
        blocks.add(
            i,
            time=rows,
            obs=rows * 10.0,
            err=np.full(n, 0.5),
            detrend=rows[:, None] + 100.0,
            **extra,
        )
        start += n
    return blocks


def test_concatenated_row_ranges_are_contiguous_and_in_config_order():
    """
    Given three files of 4, 3 and 5 observations fed to the shared
    accumulator,
    When finalize publishes the concatenated arrays,
    Then each element owns exactly one contiguous row range, the ranges are
    in config order and tile the arrays with no gap or overlap, and inst_map
    agrees with them row for row.
    """
    inst = _make(_BLOCKS_CONFIG)
    _filled_blocks(inst).finalize("rv")

    assert inst.row_ranges == [(0, 4), (4, 7), (7, 12)]
    assert inst.n_total_obs == 12
    assert inst.time.shape == (12,)
    assert inst.rv.shape == (12,)
    assert inst.err.shape == (12,)
    # Contiguous, ordered, gapless, and exactly what inst_map says.
    assert inst.row_ranges[0][0] == 0
    assert inst.row_ranges[-1][1] == inst.n_total_obs
    for i, (lo, hi) in enumerate(inst.row_ranges):
        if i:
            assert lo == inst.row_ranges[i - 1][1]
        assert np.array_equal(
            np.flatnonzero(inst.inst_map == i), np.arange(lo, hi)
        )


def test_concatenated_detrend_blocks_land_in_their_own_row_range():
    """
    Given per-file detrend columns,
    When finalize builds the block-diagonal design matrix,
    Then each file's block occupies exactly its own row range and its own
    column range -- the property _build_block_detrend relies on the loop
    order for, and which an out-of-order block would break silently.
    """
    inst = _make(_BLOCKS_CONFIG)
    _filled_blocks(inst).finalize("rv")

    assert inst.n_detrend_per_inst == [1, 1, 1]
    assert inst.total_detrend_cols == 3
    assert inst.detrend_matrix.shape == (12, 3)
    for i, (lo, hi) in enumerate(inst.row_ranges):
        block = inst.detrend_matrix[lo:hi, i]
        # whitened per (instrument, column) -- review 6.5.2
        assert np.allclose(block, _whiten(inst.time[lo:hi] + 100.0))
        # Nothing of this file leaks into another file's rows/columns.
        assert np.all(inst.detrend_matrix[:lo, i] == 0.0)
        assert np.all(inst.detrend_matrix[hi:, i] == 0.0)


def test_concatenated_side_arrays_stay_row_aligned():
    """
    Given a per-epoch side array (mulensing's observer_pos),
    When finalize concatenates it,
    Then it is published on the owner with one row per observation, aligned
    with `time` inside every element's row range.
    """
    inst = _make(_BLOCKS_CONFIG)
    _filled_blocks(inst, sides=True).finalize("flux")

    assert inst.observer_pos.shape == (12, 3)
    assert np.array_equal(inst.observer_pos[:, 0], inst.time)
    for lo, hi in inst.row_ranges:
        assert np.array_equal(
            inst.observer_pos[lo:hi, 1], inst.time[lo:hi] + 0.5
        )


def test_concatenated_data_rejects_out_of_order_files():
    """
    Given a child that adds its files out of config order,
    When add is called,
    Then it raises -- the contiguous-block invariant cannot be restored
    afterwards and would corrupt the detrend matrix and side arrays silently.
    """
    inst = _make(_BLOCKS_CONFIG)
    blocks = inst._concat_blocks()
    blocks.add(0, time=[1.0], obs=[1.0], err=[1.0])
    with pytest.raises(ValueError, match="config order"):
        blocks.add(2, time=[1.0], obs=[1.0], err=[1.0])


@pytest.mark.parametrize("bad", ["obs", "err", "detrend", "side"])
def test_concatenated_data_rejects_ragged_blocks(bad):
    """
    Given a per-file array whose length disagrees with that file's times,
    When add is called,
    Then it raises rather than concatenating a misaligned block.
    """
    inst = _make(_BLOCKS_CONFIG)
    blocks = inst._concat_blocks()
    kwargs = {
        "time": np.zeros(3),
        "obs": np.zeros(3),
        "err": np.zeros(3),
    }
    if bad == "side":
        kwargs["observer_pos"] = np.zeros((2, 3))
    elif bad == "detrend":
        kwargs["detrend"] = np.zeros((2, 1))
    else:
        kwargs[bad] = np.zeros(2)

    with pytest.raises(ValueError, match="rows but time has"):
        blocks.add(0, **kwargs)


def test_concatenated_data_rejects_a_side_array_on_only_some_files():
    """
    Given a side array supplied for the first file but not the second,
    When add is called,
    Then it raises: a per-epoch array must cover every element or none, or
    the concatenated array would silently stop being row-aligned.
    """
    inst = _make(_BLOCKS_CONFIG)
    blocks = inst._concat_blocks()
    blocks.add(
        0,
        time=np.zeros(3),
        obs=np.zeros(3),
        err=np.zeros(3),
        observer_pos=np.zeros((3, 3)),
    )
    with pytest.raises(ValueError, match="missing per-epoch array"):
        blocks.add(1, time=np.zeros(2), obs=np.zeros(2), err=np.zeros(2))


def test_concatenated_data_requires_every_element_to_contribute():
    """
    Given only two of three elements fed to the accumulator,
    When finalize runs,
    Then it raises: inst_map must address every configured instrument.
    """
    inst = _make(_BLOCKS_CONFIG)
    blocks = inst._concat_blocks()
    for i in range(2):
        blocks.add(i, time=np.zeros(3), obs=np.zeros(3), err=np.zeros(3))
    with pytest.raises(ValueError, match="contributed data blocks"):
        blocks.finalize("rv")


def test_concatenated_data_takes_detrend_columns_past_the_roles():
    """
    Given a DataFrame with columns past the three canonical roles,
    When add receives it as `df`,
    Then those columns become this file's detrend block (and a DataFrame with
    only the roles yields an empty one).
    """
    pd = pytest.importorskip("pandas")
    inst = _make(_BLOCKS_CONFIG[:2])
    blocks = inst._concat_blocks()
    df = pd.DataFrame(
        {
            0: [1.0, 2.0],
            1: [3.0, 4.0],
            2: [0.1, 0.1],
            3: [7.0, 8.0],
            4: [9.0, 10.0],
        }
    )
    blocks.add(
        0,
        time=df.iloc[:, 0].values,
        obs=df.iloc[:, 1].values,
        err=df.iloc[:, 2].values,
        df=df,
    )
    blocks.add(
        1,
        time=np.zeros(2),
        obs=np.zeros(2),
        err=np.zeros(2),
        df=df.iloc[:, :3],
    )
    blocks.finalize("rv")

    assert inst.n_detrend_per_inst == [2, 0]
    # whitened per (instrument, column) -- review 6.5.2
    assert np.allclose(inst.detrend_matrix[:2, 0], _whiten([7.0, 8.0]))
    assert np.allclose(inst.detrend_matrix[:2, 1], _whiten([9.0, 10.0]))


# ---------------------------------------------------------------------------
# Noise registration into a manifest
# ---------------------------------------------------------------------------
def test_register_noise_additive_adds_jitter_variance_and_jitter():
    """
    Given the default jitter_variance noise model,
    When _register_noise runs with a floor,
    Then the manifest gains jitter_variance and jitter, and the computed floor
    rides the "overrides" channel (clipped against the user's own bound in
    ConfigManager.resolve) rather than a plain manifest option (which would
    replace the resolved bound outright).
    """
    inst = _make([{"file": "a.dat"}])
    manifest = {}
    inst._register_noise(manifest, jittervar_lower=[-0.1])
    assert manifest["jitter_variance"] == {"overrides": {"lower": [-0.1]}}
    assert manifest["jitter"] == "default"
    assert "err_scale" not in manifest


def test_register_noise_multiplicative_adds_err_scale_only():
    """
    Given the err_scale noise model,
    When _register_noise runs,
    Then the manifest gains only err_scale (no jitter_variance/jitter).
    """
    inst = _DummyErrScale([{"file": "a.dat"}], config_manager=None)
    manifest = {}
    inst._register_noise(manifest)
    assert manifest == {"err_scale": None}


def test_register_noise_additive_requires_a_floor():
    """
    Given the additive model,
    When _register_noise is called without a jittervar_lower floor,
    Then it raises (the floor keeps the total variance positive).
    """
    inst = _make([{"file": "a.dat"}])
    with pytest.raises(ValueError):
        inst._register_noise({})


# ---------------------------------------------------------------------------
# The computed jitter floor vs the user's own bound (end to end through System)
#
# The floor is a validity limit: below -min(err)**2 the total sigma is NaN.  So
# the user may TIGHTEN jitter_variance's lower bound freely, and is clipped
# (loudly) only when asking to go below the floor.  Before the fix the floor
# rode the manifest's plain options, which replace the resolved array, so every
# user bound was discarded silently.
# ---------------------------------------------------------------------------
JITTERVAR_TO_INTERNAL = ((u.m / u.s).to(u.solRad / u.d)) ** 2
# The fixture's errors are a flat 3 m/s, so the floor is -0.95 * 3**2.
EXPECTED_FLOOR = -0.95 * 9.0


@pytest.fixture(scope="module")
def rv_file(tmp_path_factory):
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, 40))
    rv = 30.0 * np.sin(2 * np.pi * t / 17.0) + rng.normal(0, 3.0, 40)
    err = np.full(40, 3.0)
    path = tmp_path_factory.mktemp("jitter_rv") / "a.rv"
    np.savetxt(path, np.column_stack([t, rv, err]))
    return str(path)


def _jittervar_lower(rv_file, user_entry=None):
    """Resolved jitter_variance lower bound (user units, m2/s2)."""
    from exozippy.system import System

    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": [{"name": "A_inst", "file": rv_file}],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.logP": {"initval": np.log10(17.0)},
        "orbit.b.tc": {"initval": 2455010.0},
    }
    if user_entry is not None:
        params["rvinstrument.A_inst.jitter_variance"] = user_entry
    system = System(config, params)
    system.prepare()
    system.build_model()
    assert system.rvinstrument.jittervar_lower[0] == pytest.approx(
        EXPECTED_FLOOR
    )
    return system.rvinstrument.jitter_variance.lower[0] / JITTERVAR_TO_INTERNAL


def test_jitter_floor_applies_when_the_user_sets_no_bound(rv_file):
    """
    Given an RV instrument whose params file says nothing about jitter_variance,
    When the model is built,
    Then the resolved lower bound is the data-derived floor (which is tighter
    than the defaults.yaml -1000 m2/s2).
    """
    assert _jittervar_lower(rv_file) == pytest.approx(EXPECTED_FLOOR)


def test_user_lower_above_the_floor_wins(rv_file):
    """
    Given a user who explicitly sets jitter_variance lower: 0 (no negative
    jitter), which is above the computed floor,
    When the model is built,
    Then the user's bound survives instead of being replaced by the floor.
    """
    assert _jittervar_lower(rv_file, {"lower": 0.0}) == pytest.approx(0.0)


def test_user_lower_between_floor_and_default_wins(rv_file):
    """
    Given a user bound that is tighter than the floor but looser than zero,
    When the model is built,
    Then it is used verbatim -- the clip is one-sided.
    """
    assert _jittervar_lower(rv_file, {"lower": -1.0}) == pytest.approx(-1.0)


def test_user_lower_below_the_floor_is_clipped_and_warned(rv_file, caplog):
    """
    Given a user bound BELOW the computed floor, where err**2 +
    jitter_variance can go negative and the likelihood is NaN,
    When the model is built,
    Then the floor wins (it is a validity limit, not a preference) and the
    clip is reported rather than applied silently.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="exozippy.config"):
        lower = _jittervar_lower(rv_file, {"lower": -1.0e6})

    assert lower == pytest.approx(EXPECTED_FLOOR)
    assert any(
        "jitter_variance" in r.message and "validity bound" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Optional per-instrument plot styling (config, not Parameters)
# ---------------------------------------------------------------------------
def test_plot_styles_default_to_none():
    """
    Given instruments without a plot: block,
    When the base __init__ loads styles,
    Then plot_color/plot_marker are None and the trace style carries only the
    series index (so the theme default by index still applies).
    """
    inst = _make([{"file": "a.dat"}, {"file": "b.dat"}])
    assert inst.plot_color == [None, None]
    assert inst.plot_marker == [None, None]
    assert inst._data_trace_style(1) == {"series_index": 1}


def test_plot_styles_read_user_overrides():
    """
    Given an instrument with plot: {color, marker},
    When styles are loaded,
    Then the overrides surface in _data_trace_style alongside series_index.
    """
    inst = _make(
        [
            {"file": "a.dat", "plot": {"color": "#1f77b4", "marker": "s"}},
            {"file": "b.dat", "plot": {"color": "red"}},
            {"file": "c.dat"},
        ]
    )
    assert inst.plot_color == ["#1f77b4", "red", None]
    assert inst.plot_marker == ["s", None, None]
    assert inst._data_trace_style(0) == {
        "series_index": 0,
        "color": "#1f77b4",
        "marker": "s",
    }
    assert inst._data_trace_style(1) == {"series_index": 1, "color": "red"}
    assert inst._data_trace_style(2) == {"series_index": 2}


def test_plot_style_schema_entry_shape():
    """
    Given the shared plot config-schema entry,
    When inspected,
    Then it declares the standard option shape consumed by introspection.
    """
    entry = Instrument._plot_style_config_schema()
    assert entry["key"] == "plot"
    assert entry["kind"] == "option"
    assert entry["required"] is False
    assert set(entry) >= {"key", "kind", "accepts", "required", "doc"}


# ---------------------------------------------------------------------------
# Trace.style serialization (plumbing target for the plot styles)
# ---------------------------------------------------------------------------
def test_trace_style_serializes_when_present():
    """
    Given a data Trace carrying a style dict,
    When to_json runs,
    Then the style is included (and absent when None).
    """
    t = Trace(
        name="TESS",
        role="data",
        kind="scatter",
        x=[1.0],
        y=[2.0],
        style={"series_index": 0, "marker": "s"},
    )
    d = t.to_json()
    assert d["style"] == {"series_index": 0, "marker": "s"}

    plain = Trace(name="m", role="model", kind="line", x=[1.0], y=[2.0])
    assert "style" not in plain.to_json()


def test_rows_is_the_published_row_range():
    """
    Given the shared accumulator's published row_ranges,
    When Instrument.rows(i) is asked for an element's rows,
    Then it is that contiguous slice, and it selects exactly what the
    inst_map boolean scan selects -- the two spellings the class docstring
    says are equivalent.
    """
    inst = _make(_BLOCKS_CONFIG)
    _filled_blocks(inst).finalize("rv")

    for i, (lo, hi) in enumerate(inst.row_ranges):
        sl = inst.rows(i)
        assert (sl.start, sl.stop) == (lo, hi)
        np.testing.assert_array_equal(
            inst.time[sl], inst.time[inst.inst_map == i]
        )


def test_rows_refuses_a_component_that_never_concatenated():
    """
    Given a component with no row_ranges (it keeps per-file datasets),
    When rows(i) is called,
    Then it raises saying so, rather than silently selecting nothing.
    """
    inst = _make(_BLOCKS_CONFIG)

    with pytest.raises(ValueError, match="no row_ranges"):
        inst.rows(0)
