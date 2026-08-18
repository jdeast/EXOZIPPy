"""Two data-derived start values that used to bypass the provenance pipeline.

Both were special cases that reached the model without ever acquiring a rank,
and both are now ordinary citizens of the machinery described in config.py:

  1. ``transit.baseline`` (per-file median flux) was a plain manifest OPTION.
     Options are merged as ``{**cfg, **options}`` AFTER
     ``ConfigManager.resolve``, so they beat the user's params file outright.
     A data-derived START value has exactly the wrong precedence that way --
     an explicit ``transit.<name>.baseline`` (a restart file, or the
     EXOFASTv2 solution in ``examples/gj1214``) was silently discarded.  It
     is now a ``config_manager.add_hint`` at RANK_DERIVED_DATA: above the
     defaults.yaml 1.0, below the user.

  2. ``mulensinstrument.zeropoint`` never became a ``Parameter`` at all --
     ``_build_sed_flux_constraint`` resolved the config block itself and read
     only ``mu``/``sigma``, so an ``initval`` in a params file did nothing.
     It is now a DERIVED manifest parameter (its value is fixed by f_source
     and the SED; see the docstring there for why derived and not sampled),
     which means ``resolve()``'s "user gave mu, start there" rule applies to
     it like it does to everything else.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.config import RANK_DEFAULT, RANK_DERIVED_DATA, RANK_USER
from exozippy.system import System

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"


# ---------------------------------------------------------------------------
# transit.baseline
# ---------------------------------------------------------------------------

# Two flat light curves at deliberately non-unity levels, so the median is
# distinguishable from the defaults.yaml 1.0 AND from each other.
_LEVELS = (0.75, 1.25)


def _write_lc(path, level):
    t = 2459634.0 + np.linspace(-0.2, 0.2, 41)
    flux = np.full_like(t, level)
    err = np.full_like(t, 1e-3)
    np.savetxt(path, np.column_stack([t, flux, err]))
    return str(path)


def _transit_config(tmp_path):
    files = [
        _write_lc(tmp_path / f"lc{i}.dat", lvl)
        for i, lvl in enumerate(_LEVELS)
    ]
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "V", "filter": "V", "ld_law": "quadratic"}],
        "transit": [
            {"name": f"inst{i}", "file": f, "band": "V"}
            for i, f in enumerate(files)
        ],
    }


def _transit_params():
    return {
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.b.period": {"initval": 3.0},
        "orbit.b.tc": {"initval": 2459634.3},
        "planet.b.radius": {"initval": 1.0},
    }


def _prepared(config, user_params):
    system = System(config, user_params)
    system.prepare()
    return system


def test_baseline_is_a_hint_at_rank_derived_data(tmp_path):
    """
    Given two transit light curves at non-unity flux levels,
    When the system is prepared,
    Then each file's median flux is registered as a ConfigManager HINT at
    RANK_DERIVED_DATA -- not smuggled in as a manifest option, which would
    carry no rank at all.
    """
    system = _prepared(_transit_config(tmp_path), _transit_params())
    cm = system.config_manager

    for i, level in enumerate(_LEVELS):
        path = f"transit.{i}.baseline"
        assert path in cm.hints, f"{path} was never hinted"
        assert cm.hints[path] == pytest.approx(level)
        assert cm.hint_ranks[path] == RANK_DERIVED_DATA
        assert RANK_DEFAULT < cm.hint_ranks[path] < RANK_USER


def test_baseline_manifest_entry_carries_no_initval_option(tmp_path):
    """
    Given the transit component after stage 3,
    When its manifest entry for baseline is inspected,
    Then it carries no 'initval' option -- the value travels through the
    hint channel now, and a manifest option would override the resolved
    config (and so the user) at build time.
    """
    system = _prepared(_transit_config(tmp_path), _transit_params())
    entry = system.transit.manifest["baseline"] or {}
    assert "initval" not in entry


def test_resolved_baseline_equals_the_per_file_median(tmp_path):
    """
    Given no user entry for baseline,
    When the model is built,
    Then each element's start value is that file's own median flux, i.e. the
    hint won over the defaults.yaml 1.0.
    """
    system = _prepared(_transit_config(tmp_path), _transit_params())
    system.build_model()

    initvals = np.atleast_1d(system.transit.baseline.initval)
    assert initvals == pytest.approx(np.array(_LEVELS))


def test_user_baseline_beats_the_data_derived_hint(tmp_path):
    """
    Given a params file that names transit.inst0.baseline explicitly,
    When the model is built,
    Then the user's value is what the parameter starts at (RANK_USER 100
    beats RANK_DERIVED_DATA 60), while the file the user said nothing about
    still gets its median.

    This is the actual improvement: as a manifest option the median
    overwrote the user's number after resolve() had already honored it.
    """
    user_value = 0.5
    params = _transit_params()
    params["transit.inst0.baseline"] = {"initval": user_value}

    system = _prepared(_transit_config(tmp_path), params)
    system.build_model()

    initvals = np.atleast_1d(system.transit.baseline.initval)
    assert initvals[0] == pytest.approx(user_value)
    assert initvals[1] == pytest.approx(_LEVELS[1])


def test_user_baseline_is_reported_as_user_provenance(tmp_path):
    """
    Given the same explicit user entry,
    When the solution is exported,
    Then the user's element is labelled 'user' and the untouched one 'data'.
    A manifest option produced neither label -- it never entered the
    provenance ledger at all.
    """
    params = _transit_params()
    params["transit.inst0.baseline"] = {"initval": 0.5}
    system = _prepared(_transit_config(tmp_path), params)

    export = system.config_manager.export_solution(
        derived_params=system.derived_params()
    )["parameters"]
    assert export["transit.inst0.baseline"]["provenance"]["label"] == "user"
    assert export["transit.inst1.baseline"]["provenance"]["label"] == "data"


def test_baseline_falls_back_to_one_without_data(tmp_path):
    """
    Given a transit component whose load_data never ran (no measured
    medians),
    When register_parameters pushes its hints,
    Then nothing is hinted and the defaults.yaml 1.0 stands -- the same
    value load_data seeds baseline_init with, so the no-data fallback is
    unchanged by the move to the hint channel.
    """
    system = System(_transit_config(tmp_path), _transit_params())
    transit = system.transit
    # Simulate "load_data never measured anything" without breaking the rest
    # of prepare(): _hint_baseline is the only reader of baseline_init.
    assert not hasattr(transit, "baseline_init")
    transit._hint_baseline()  # must not raise
    assert not [k for k in system.config_manager.hints if "baseline" in k]

    cfg = system.config_manager.resolve(
        "transit", "baseline", shape=(transit.n_elements,), names=transit.names
    )
    assert np.atleast_1d(cfg["initval"]) == pytest.approx(
        np.ones(transit.n_elements)
    )


def test_non_finite_median_is_skipped_not_hinted(tmp_path, caplog):
    """
    Given a light curve whose median flux is not finite,
    When the hints are pushed,
    Then that element is skipped with a warning and keeps the defaults.yaml
    start -- a NaN hint would propagate into the relaxation engine.
    """
    system = System(_transit_config(tmp_path), _transit_params())
    transit = system.transit
    transit.baseline_init = [np.nan, _LEVELS[1]]

    with caplog.at_level("WARNING"):
        transit._hint_baseline()

    hints = system.config_manager.hints
    assert "transit.0.baseline" not in hints
    assert hints["transit.1.baseline"] == pytest.approx(_LEVELS[1])
    assert "median flux is not finite" in caplog.text


# ---------------------------------------------------------------------------
# mulensinstrument.zeropoint
# ---------------------------------------------------------------------------


def _kmt_inputs():
    with open("KMT-2019-BLG-1806.yaml") as f:
        config = yaml.safe_load(f)
    with open(config["parameter_file"]) as f:
        user_params = yaml.safe_load(f)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    return config, user_params


def _build_kmt(user_overrides=None):
    """Build the KMT-2019-BLG-1806 example, optionally with extra params."""
    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")
    cwd = os.getcwd()
    os.chdir(_KMT_DIR)
    try:
        config, user_params = _kmt_inputs()
        user_params.update(user_overrides or {})
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


@pytest.fixture(scope="module")
def kmt_default():
    return _build_kmt()


def test_zeropoint_is_a_derived_parameter(kmt_default):
    """
    Given a mulensing topology with a sed: block,
    When the model is built,
    Then zeropoint is a real Parameter -- in the manifest, derived (not
    sampled: f_source and the SED determine it exactly), and therefore
    absent from the free variables.
    """
    from exozippy.components.parameter import Parameter

    system, model = kmt_default
    inst = system.mulensinstrument

    assert "zeropoint" in inst.manifest
    assert isinstance(inst.zeropoint, Parameter)
    assert ("mulensinstrument", "zeropoint") in system.derived_params()
    free = [rv.name for rv in model.free_RVs]
    assert not any("zeropoint" in n for n in free)


def test_zeropoint_initval_defaults_to_mu(kmt_default):
    """
    Given no user entry, so mu comes from defaults.yaml (0 +/- 0.2 mag),
    When the resolved config is read,
    Then the prior center is 0 -- the value the Gaussian is applied against
    by Parameter.build_pymc's derived-with-sigma branch.
    """
    system, _ = kmt_default
    inst = system.mulensinstrument
    assert np.atleast_1d(inst.zeropoint.mu) == pytest.approx(
        np.zeros(inst.n_elements)
    )
    assert np.atleast_1d(inst.zeropoint.sigma) == pytest.approx(
        np.full(inst.n_elements, 0.2)
    )


def test_zeropoint_honours_a_user_mu_as_its_start_and_center():
    """
    Given a params file naming mulensinstrument.KMTC04.zeropoint with a mu,
    When the model is built,
    Then that element's prior center AND its start value are the user's mu
    -- the same "mu is the start when no initval is given" rule every other
    parameter follows.  Reading only mu out of the config block by hand, as
    the old code did, could never honour an initval at all.
    """
    system, _ = _build_kmt(
        {"mulensinstrument.KMTC04.zeropoint": {"mu": 21.0, "sigma": 0.05}}
    )
    zp = system.mulensinstrument.zeropoint
    i = list(system.mulensinstrument.names).index("KMTC04")

    assert np.atleast_1d(zp.mu)[i] == pytest.approx(21.0)
    assert np.atleast_1d(zp.sigma)[i] == pytest.approx(0.05)
    assert np.atleast_1d(zp.initval)[i] == pytest.approx(21.0)


def test_zeropoint_honours_an_explicit_user_initval():
    """
    Given a params file that gives zeropoint an explicit initval,
    When the model is built,
    Then the Parameter carries it.  Before, nothing read this key: the
    constraint was built from mu/sigma alone, so an initval in a params file
    was silently inert -- exactly the "special case that ignores initval"
    this change removes.
    """
    system, _ = _build_kmt(
        {"mulensinstrument.KMTC04.zeropoint": {"initval": 17.5}}
    )
    zp = system.mulensinstrument.zeropoint
    i = list(system.mulensinstrument.names).index("KMTC04")
    assert np.atleast_1d(zp.initval)[i] == pytest.approx(17.5)


def test_zeropoint_units_go_through_the_generic_conversion():
    """
    Given a user who states the zeropoint prior in millimagnitudes,
    When the model is built,
    Then the value is converted to the internal mag unit (a factor 1e-3),
    because it is now an ordinary Parameter.  A hand-resolved config block
    read the raw number in whatever unit the user happened to write.
    """
    system, _ = _build_kmt(
        {
            "mulensinstrument.KMTC04.zeropoint": {
                "mu": 500.0,
                "sigma": 20.0,
                "unit": "mmag",
            }
        }
    )
    zp = system.mulensinstrument.zeropoint
    i = list(system.mulensinstrument.names).index("KMTC04")
    assert np.atleast_1d(zp.mu)[i] == pytest.approx(0.5)
    assert np.atleast_1d(zp.sigma)[i] == pytest.approx(0.02)
