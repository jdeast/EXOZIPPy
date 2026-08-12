"""Regressions for the "silent bandaid" class of defect.

Every test here pins a place that used to swallow a failure and continue
with a value that was WRONG rather than absent -- a wrong number, not a
missing one -- and asserts that the code is now loud instead.  They are
collected in one file because the defect class, not the module, is what
they have in common; each test names its own site.
"""

import importlib
import logging

import numpy as np
import pytest

from exozippy.components import factory
from exozippy.components.sed.bc_grid import _load_alias_table

# ---------------------------------------------------------------------------
# factory.py / system.py -- a component whose module fails to import must not
# be quietly dropped from a config that asks for it.
# ---------------------------------------------------------------------------


def test_config_naming_an_unimportable_component_raises(monkeypatch):
    """
    Given a component module that raises ImportError (a missing optional
      dependency, e.g. VBMicrolensing for mulensing/op.py),
    When a config names that component's YAML key,
    Then System raises, naming the module and the underlying ImportError.

    Pre-fix: discover_components logged a warning and dropped the
    component, then System warned "YAML key 'lens' does not match any
    registered component and will be ignored" -- which reads like a typo --
    and fitted a model with no lens at all, to completion.
    """
    # ARRANGE -- make exactly the lens module unimportable
    real_import = importlib.import_module

    def fake_import(name, *a, **kw):
        if name.endswith(".mulensing.lens"):
            raise ImportError("no module named 'vbmicrolensing'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(factory.importlib, "import_module", fake_import)

    from exozippy.system import System

    config = {
        "star": [{"name": "Lens"}],
        "lens": [{"name": "L", "lenses": ["star.0"], "sources": ["star.0"]}],
    }

    # ACT / ASSERT
    with pytest.raises(ImportError) as exc:
        System(config, {})

    msg = str(exc.value)
    assert "lens" in msg
    assert "vbmicrolensing" in msg.lower(), (
        f"the error must carry the underlying ImportError, got: {msg}"
    )


def test_unknown_yaml_key_still_only_warns(caplog):
    """
    Given a YAML key that matches no component and never failed to import,
    When System is constructed,
    Then it still only warns -- the raise above must not swallow the
    ordinary typo case.
    """
    # ARRANGE / ACT
    from exozippy.system import System

    with caplog.at_level(logging.WARNING):
        System({"star": [{"name": "A"}], "not_a_component": [{}]}, {})

    # ASSERT
    assert any("not_a_component" in r.message for r in caplog.records), (
        "the typo case must still warn"
    )


# ---------------------------------------------------------------------------
# components/component.py -- a dependency that NAMES its index map must not
# silently fall back to the unsliced vector.
# ---------------------------------------------------------------------------


def test_named_dep_map_that_does_not_exist_raises():
    """
    Given a lens whose manifest deps name 'lens_map' but whose lens_map is
      gone (a build_maps bug, or a dep renamed without its map),
    When build_model wires the expressions,
    Then it raises naming the dependency and the missing map.

    Pre-fix: the WHOLE star.pm_ra / star.mass vector was passed to the
    physics function in place of the requested elements.  Where the lengths
    happened to match, it broadcast silently and paired the wrong bodies --
    another star's mass into this lens's theta_E, with a healthy logp.
    """
    # ARRANGE
    from exozippy.system import System

    config = {
        "star": [
            {"name": "Lens", "mist": False},
            {"name": "Source", "mist": False},
        ],
        "lens": [
            {"name": "Lens", "lenses": ["star.0"], "sources": ["star.1"]}
        ],
    }
    system = System(config, {})
    system.prepare()
    assert hasattr(system.lens, "lens_map")
    del system.lens.lens_map

    # ACT / ASSERT
    with pytest.raises(AttributeError) as exc:
        system.build_model()

    msg = str(exc.value)
    assert "lens_map" in msg and "names the index map" in msg


# ---------------------------------------------------------------------------
# components/rm.py -- "no star in the primary group" was silently "star 0".
# ---------------------------------------------------------------------------


def test_rm_primary_star_index_raises_when_the_group_has_no_star():
    """
    Given an orbit whose primary body group contains no star,
    When the RM model looks up the transited star,
    Then it raises instead of defaulting to star 0.

    Pre-fix `next(..., 0)` handed star 0's vmacro / vbeta / vmicro to the
    Hirano broadening kernel, biasing vsini and lambda -- the two numbers
    an RM fit exists to measure -- in any multi-star system.
    """
    # ARRANGE
    import types

    from exozippy.components.rm import rm_primary_star_index

    orbit = types.SimpleNamespace(
        primary_bodies={0: [("star", 2), ("planet", 0)], 1: [("planet", 0)]}
    )

    # ACT / ASSERT -- the good case still resolves the RIGHT star, not 0
    assert rm_primary_star_index(orbit, 0) == 2

    with pytest.raises(ValueError, match=r"no star in its primary body group"):
        rm_primary_star_index(orbit, 1)


# ---------------------------------------------------------------------------
# sed/filters/filternames.txt -- an alias that appears in two rows resolves
# to whichever row happens to come first.
# ---------------------------------------------------------------------------


def test_no_duplicate_filter_aliases_outside_claret():
    """
    Given the shipped filter alias table,
    When each column's non-placeholder values are counted,
    Then no name appears in more than one row.

    resolve_filter_name matches a user string against EVERY column and
    takes .values[0], so a duplicate silently resolves to whichever row is
    first.  'iPS' was a copy-paste typo on the PAN-STARRS w row, which made
    'wPS' unreachable and put PS1.w one row-order change away from being
    served for a request for PS1.i.

    The Claret column is EXCLUDED deliberately: its bare 'I' and 'R' really
    do appear on both the Bessell and the Cousins rows, and several shipped
    examples use `filter: "I"`, so disambiguating that is a separate
    (science) decision -- see the audit report.
    """
    # ARRANGE
    df = _load_alias_table()
    cols = [c for c in df.columns if c != "Claret"]

    # ACT
    dupes = {}
    for col in cols:
        for value in df[col].dropna():
            name = str(value).strip()
            if not name or name.lower() in ("unsupported", "nan", "-"):
                continue
            rows = df.index[df[cols].eq(name).any(axis=1)].tolist()
            if len(rows) > 1:
                dupes[name] = rows

    # ASSERT
    assert not dupes, f"filter names resolving to more than one row: {dupes}"


# ---------------------------------------------------------------------------
# outputs/ledger.py -- the internal->user unit factor was replaced by 1.0
# for every multi-element parameter.
# ---------------------------------------------------------------------------


def test_ledger_unit_factor_guard_allows_scalar_broadcast():
    """
    Given a parameter whose `unit:` is a scalar string (so
      _get_conversion_factors returns ONE factor) but whose value vector has
      several elements,
    When the seed ledger converts internal values to user units,
    Then the single factor is used for every element.

    Pre-fix the guard was `if factors.size != p0.size: factors =
    np.ones(...)`, which fired for EVERY vector parameter -- reporting
    radians under a 'deg' label (57.3x) for orbit.omega / inc / bigomega /
    lam and star.ra / dec as soon as a system had two orbits or two stars.
    """
    # ARRANGE
    import inspect

    from exozippy.outputs import ledger

    src = inspect.getsource(ledger)

    # ASSERT
    assert "factors = np.ones(p0.size)" not in src, (
        "the identity-substitution fallback is back; a size-1 factor "
        "broadcasts correctly and must not be replaced by ones"
    )
    assert "factors.size not in (1, p0.size)" in src


# ---------------------------------------------------------------------------
# samplers/ptde.py -- a never-proposed rung pair was scored as a
# 100%-rejecting link.
# ---------------------------------------------------------------------------


def test_ladder_health_ignores_never_proposed_pairs(caplog):
    """
    Given a 4-rung ladder where the middle pair was never PROPOSED
      (routine: DEO alternates parities and the counters reset every
      adaptation window),
    When ladder_health_report measures the communication barrier,
    Then the unmeasured pair is interpolated, not counted as a full
    rejection.

    Pre-fix `1 - acc/maximum(prop, 1)` gave that pair r_k = 1.0, inflating
    Lambda and firing the "PT ladder is communication-limited, raise
    n_temps" warning -- the number this project sizes its ladders on -- for
    a ladder that was mixing fine.
    """
    # ARRANGE -- 3 pairs; the middle one has zero proposals, the two
    # measured ones reject 20% of the time.
    from exozippy.samplers.ptde import ladder_health_report

    temperatures = [1.0, 2.0, 4.0, 8.0]
    n_swap_propose = np.array([100.0, 0.0, 100.0])
    n_swap_accept = np.array([80.0, 0.0, 80.0])

    # ACT
    with caplog.at_level(logging.WARNING):
        lam = ladder_health_report(temperatures, n_swap_accept, n_swap_propose)

    # ASSERT -- 0.2 + interp(0.2) + 0.2, not 0.2 + 1.0 + 0.2
    assert lam == pytest.approx(0.6, abs=1e-9), (
        f"Lambda={lam}; 1.4 means the unproposed pair was scored as a "
        f"100%-rejecting link"
    )
    assert not [
        r for r in caplog.records if "communication-limited" in r.message
    ], "a healthy ladder must not be reported as communication-limited"


# ---------------------------------------------------------------------------
# samplers/_common.py -- a logp that RAISES became a proposal with zero
# posterior density, silently.
# ---------------------------------------------------------------------------


def test_worker_logp_exception_is_reported_once_per_type(caplog):
    """
    Given a logp function that raises (a celerite Cholesky failure, a
      magnification backend error, ...),
    When a sampler worker evaluates a proposal,
    Then it still returns -inf, but logs the exception type, message and
    proposal once -- and does not spam on repeats.

    -inf is not "absent": both Metropolis tests read it as zero posterior
    density, so a region that merely ERRORS is excluded from the posterior
    exactly as if the model had ruled it out.  Pre-fix this was a bare
    `except Exception: return -np.inf` with no counter, no warning and no
    sample_stats entry, and it swallowed the exception before
    ptde_async's error_callback could fire.
    """
    # ARRANGE
    from exozippy.samplers import _common

    def boom(_):
        raise ValueError("celerite cholesky failed")

    saved_fn = _common._PTDE_LOGP_FN
    saved_seen = set(_common._LOGP_EXC_SEEN)
    _common._PTDE_LOGP_FN = boom
    _common._LOGP_EXC_SEEN.clear()
    try:
        # ACT
        with caplog.at_level(logging.ERROR):
            first = _common._eval_logp({"x_raw": np.array([1.0, 2.0])})
            second = _common._eval_logp({"x_raw": np.array([3.0])})
    finally:
        _common._PTDE_LOGP_FN = saved_fn
        _common._LOGP_EXC_SEEN.clear()
        _common._LOGP_EXC_SEEN.update(saved_seen)

    # ASSERT
    assert first == -np.inf and second == -np.inf
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, (
        f"expected exactly one report per exception type, got {len(errors)}"
    )
    msg = errors[0].message
    assert "ValueError" in msg and "celerite cholesky failed" in msg
    assert "x_raw" in msg


# ---------------------------------------------------------------------------
# ephemeris.py -- the extrapolation warning went nowhere and misreported
# which end was out of range.
# ---------------------------------------------------------------------------


def test_ephemeris_extrapolation_warning_reaches_the_log_and_names_the_end(
    tmp_path, caplog
):
    """
    Given epochs ABOVE the ephemeris grid's last time,
    When interpolate_ephemeris extrapolates,
    Then it logs a warning naming the grid span and the offending (max)
    epoch.

    Pre-fix the message went through warnings.warn -- deduplicated, and
    since nothing calls logging.captureWarnings it never reached
    <prefix>.log, the file users keep -- and it printed
    `Requested: {np.min(time)}`, so an over-range epoch was reported with
    an IN-range number.  The value at stake is a cubic-extrapolated
    observer position feeding microlensing pi_E and astrometric parallax
    factors.
    """
    # ARRANGE -- a tiny 5-point ephemeris covering 2458000-2458004
    from exozippy.ephemeris import interpolate_ephemeris

    path = tmp_path / "toy.eph"
    t = np.arange(2458000.0, 2458005.0)
    np.savetxt(
        path, np.column_stack([t, np.cos(t), np.sin(t), np.zeros_like(t)])
    )

    # ACT -- ask for an epoch past the END of the grid
    with caplog.at_level(logging.WARNING):
        interpolate_ephemeris(np.array([2458002.0, 2458010.0]), str(path))

    # ASSERT
    messages = " ".join(r.message for r in caplog.records)
    assert "outside the ephemeris range" in messages
    assert "2458010" in messages, (
        f"the out-of-range END must be named; got: {messages}"
    )


# ---------------------------------------------------------------------------
# utilities/mkticsed.py -- three fabricated numbers.
# ---------------------------------------------------------------------------


def test_gaia_zero_point_tables_are_loaded():
    """
    Given the gaiadr3-zeropoint package,
    When mkticsed is imported,
    Then zpt.load_tables() has run and get_zpt returns a real correction
    for numpy scalars.

    Pre-fix, get_zpt raised "The table of coefficients have not been
    initialized!!" on EVERY call (load_tables was never called anywhere in
    the tree) and -- once that was fixed -- TypeError under NEP 50 for the
    Python floats mkticsed passed.  Both were swallowed into the note
    string "correction failed; using raw", so the Lindegren+2021 zero point
    (~0.02-0.05 mas) was never once applied and the UNCORRECTED parallax
    became the star's distance PRIOR in the params file.
    """
    # ARRANGE / ACT
    from zero_point import zpt

    from exozippy.utilities import mkticsed  # noqa: F401  (import side effect)

    zp = float(
        zpt.get_zpt(
            np.float64(12.0),
            np.float64(1.5),
            np.float64(1.4),
            np.float64(20.0),
            np.int64(31),
        )
    )

    # ASSERT
    assert np.isfinite(zp) and zp != 0.0, (
        f"expected a real Lindegren+2021 zero point, got {zp}"
    )


def test_gaia_zero_point_failure_raises_instead_of_using_raw_parallax():
    """
    Given a get_zpt call that fails for an unexpected reason,
    When mkticsed builds the Gaia DR3 distance prior,
    Then it raises rather than writing a prior from the raw parallax.
    """
    # ARRANGE
    import inspect

    from exozippy.utilities import mkticsed

    src = inspect.getsource(mkticsed)

    # ASSERT
    assert 'zp_msg = "correction failed; using raw"' not in src, (
        "the silent fall-through to an uncorrected parallax is back"
    )
    assert "Refusing to " in src and "uncorrected " in src


def test_stromgren_photometry_requires_real_uncertainties():
    """
    Given a Stromgren catalog row with no published uncertainty,
    When mkticsed builds the SED entries,
    Then the row is skipped, as it is in every other photometry block.

    Pre-fix an ABSENT uncertainty became `max(0.01, ...)` = 0.01 mag --
    near the best achievable in Stromgren V -- so a catalog row with no
    error at all was given the tightest error in the file and dominated the
    SED likelihood.
    """
    # ARRANGE
    import inspect

    from exozippy.utilities import mkticsed

    src = inspect.getsource(mkticsed)

    # ASSERT -- the substitution was `max(0.01, <err> if isfinite else <lit>)`
    for call in (
        "max(0.01, evmag if",
        "max(0.02, eby if",
        "max(0.02, em1 if",
        "max(0.02, ec1 if",
    ):
        assert call not in src, f"fabricated Stromgren uncertainty: {call}"
    # and the finiteness of every uncertainty is now a precondition
    for guard in (
        "and np.isfinite(evmag)",
        "and np.isfinite(eby)",
        "and np.isfinite(em1)",
        "and np.isfinite(ec1)",
    ):
        assert guard in src, f"missing precondition: {guard}"


def test_schlegel_av_does_not_guess_a_column():
    """
    Given an IRSA dust table with none of the four known E(B-V) columns,
    When schlegel_av reads it,
    Then it returns None (and warns) instead of returning the first column
    that happens to parse in (0, 50) times 3.1.

    That value is written as a HARD upper bound on star.av -- the logit
    transform truncates the posterior there -- under a note asserting
    Schlegel+1998 provenance it would not have.
    """
    # ARRANGE
    from exozippy.utilities import mkticsed

    class _FakeTable:
        colnames = ["ra", "dec", "ext SandF mean"]

        def __getitem__(self, key):
            return {"ra": [12.3], "dec": [4.5], "ext SandF mean": [0.7]}[key]

    class _FakeIrsa:
        @staticmethod
        def get_query_table(coord, section=None):
            return _FakeTable()

    orig = mkticsed.IrsaDust
    mkticsed.IrsaDust = _FakeIrsa
    try:
        # ACT
        with pytest.warns(UserWarning, match="E\\(B-V\\) columns"):
            av = mkticsed.schlegel_av(10.0, 20.0)
    finally:
        mkticsed.IrsaDust = orig

    # ASSERT
    assert av is None, (
        f"expected no Av limit, got {av} -- 12.3*3.1 or 0.7*3.1 means a "
        f"column was guessed"
    )
