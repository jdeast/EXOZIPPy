"""Tests verifying the Option B unit-conversion pipeline:

After sampling, idata.posterior is converted in-place to user units by
_convert_posterior_to_user_units().  Every downstream consumer — compute_summary,
generate_posterior, and get_draws — must therefore handle user-unit inputs
and produce user-unit (or correctly converted internal-unit) outputs.

These four tests guard each link in that chain.
"""

import numpy as np
import pytest
import pymc as pm
import xarray as xr
import arviz as az
import astropy.units as u

from exozippy.components.parameter import Parameter
from exozippy.run import _convert_posterior_to_user_units, get_draws

MJ_TO_MSUN = float(u.jupiterMass.to(u.solMass))   # ≈ 9.546e-4
MSUN_TO_MJ = 1.0 / MJ_TO_MSUN                     # ≈ 1047.6


def _mass_param(**kwargs):
    """Return a planet-mass Parameter with unit=jupiterMass, internal_unit=solMass."""
    defaults = dict(
        label="planet.b.mass",
        unit=u.jupiterMass,
        internal_unit=u.solMass,
        initval=1.0,
        init_scale=0.1,
        lower=0.0,
        upper=10.0,
    )
    defaults.update(kwargs)
    return Parameter(**defaults)


def _make_idata(var_values: dict, n_chains=1, n_draws=4):
    """Minimal ArviZ InferenceData with posterior only."""
    data_vars = {}
    for name, val in var_values.items():
        arr = np.full((n_chains, n_draws), float(val))
        data_vars[name] = xr.DataArray(arr, dims=["chain", "draw"])
    return az.from_dict({"posterior": xr.Dataset(data_vars)})


# ──────────────────────────────────────────────────────────────────────────────
# Test 1  –  _convert_posterior_to_user_units
# ──────────────────────────────────────────────────────────────────────────────

def test_convert_posterior_multiplies_by_internal_to_user_factor():
    """
    Given an idata whose posterior contains planet.b.mass in internal units
    (solMass) and a param_lookup with the corresponding Parameter,
    When _convert_posterior_to_user_units is called,
    Then the posterior values are multiplied by the internal→user factor and
    are now in jupiterMass.
    """
    # ARRANGE
    internal_value = MJ_TO_MSUN          # 1.0 jupiterMass expressed in solMass
    idata = _make_idata({"planet.b.mass": internal_value})
    p = _mass_param()
    param_lookup = {p.label: p}

    # ACT
    _convert_posterior_to_user_units(idata, param_lookup)
    result = float(idata.posterior["planet.b.mass"].mean())

    # ASSERT — should now be in jupiterMass (≈ 1.0), not solMass (≈ 9.55e-4)
    assert np.isclose(result, 1.0, rtol=1e-3), (
        f"After conversion expected ~1.0 jupiterMass; got {result:.6f}. "
        f"If ≈{MJ_TO_MSUN:.4e} the conversion was not applied."
    )


def test_convert_posterior_skips_raw_variables():
    """
    Given an idata whose posterior contains both planet.b.mass and
    planet.b.mass_raw, When _convert_posterior_to_user_units is called,
    Then the _raw variable is left unchanged (whitened space, no unit meaning).
    """
    # ARRANGE
    raw_value = 0.0
    idata = _make_idata({"planet.b.mass": MJ_TO_MSUN,
                         "planet.b.mass_raw": raw_value})
    p = _mass_param()
    param_lookup = {p.label: p}

    # ACT
    _convert_posterior_to_user_units(idata, param_lookup)

    # ASSERT
    raw_result = float(idata.posterior["planet.b.mass_raw"].mean())
    assert raw_result == raw_value, (
        f"_raw variable should be unchanged; got {raw_result}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2  –  compute_summary
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_summary_treats_posterior_as_already_user_units():
    """
    Given a Parameter with unit=jupiterMass, internal_unit=solMass and
    a posterior set to values already in user units (~1.0 jupiterMass),
    When compute_summary is called,
    Then the returned median is ~1.0 (user units), not ~1048 (double-converted).

    This verifies that the from_internal() call was removed from compute_summary
    after Option B moved the unit conversion into the trace post-processor.
    """
    # ARRANGE
    p = _mass_param()
    p.posterior = np.full(200, 1.0)    # 200 samples at 1.0 jupiterMass (user units)

    # ACT
    summary = p.compute_summary()

    # ASSERT
    assert np.isclose(summary.median, 1.0, rtol=0.05), (
        f"Median should be ~1.0 jupiterMass; got {summary.median:.4f}. "
        f"If ~{MSUN_TO_MJ:.0f}, from_internal is still being applied (double-convert)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3  –  generate_posterior with param_lookup
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_posterior_with_param_lookup_round_trips_user_units():
    """
    Given a derived parameter whose expression doubles the internal-unit input
    mass, and a posterior bundle containing the input mass in user units,
    When generate_posterior is called with param_lookup,
    Then inputs are converted user→internal before PyTensor evaluation and
    the output is converted internal→user, so the result is 2× the input
    (in user units: jupiterMass).
    """
    # ARRANGE
    with pm.Model():
        p_mass = _mass_param()
        p_mass.build_pymc()

        # Derived: expression doubles the mass in the graph (internal units).
        # p_mass.value is the Deterministic("planet.b.mass") node; its name
        # matches the posterior key so generate_posterior can find it.
        p_double = Parameter(
            label="planet.b.mass2",
            unit=u.jupiterMass,
            internal_unit=u.solMass,
            expression=lambda: p_mass.value * 2.0,
        )
        p_double.build_pymc()

    # Posterior has p_mass in USER units: 1.0 jupiterMass.
    # Shape (1,): the sample dimension is last (az.extract convention).
    posterior = {p_mass.label: np.array([1.0])}
    param_lookup = {p_mass.label: p_mass}

    # ACT
    result = p_double.generate_posterior(posterior, param_lookup=param_lookup)

    # ASSERT
    # Input:   1.0 jupiterMass
    # Internal: 1.0 / MSUN_TO_MJ ≈ 9.546e-4 solMass
    # Expression: × 2 → 1.909e-3 solMass
    # Output:  1.909e-3 × MSUN_TO_MJ ≈ 2.0 jupiterMass
    result_val = float(np.squeeze(result))
    assert np.isclose(result_val, 2.0, rtol=1e-3), (
        f"Expected 2.0 jupiterMass; got {result_val:.6f}. "
        f"If ≈{2 * MJ_TO_MSUN:.2e}, output was left in solMass (no output conversion). "
        f"If ≈{2.0 * MSUN_TO_MJ:.0f}, input was not converted (double user-unit factor)."
    )


def test_generate_posterior_without_param_lookup_returns_internal_units():
    """
    Given the same setup with param_lookup=None (backward-compat path used by
    inspect_start, which passes an internal-unit bundle),
    When generate_posterior is called without param_lookup,
    Then no unit conversion is applied — inputs and output stay internal.
    """
    # ARRANGE
    with pm.Model():
        p_mass = _mass_param()
        p_mass.build_pymc()

        p_double = Parameter(
            label="planet.b.mass2",
            unit=u.jupiterMass,
            internal_unit=u.solMass,
            expression=lambda: p_mass.value * 2.0,
        )
        p_double.build_pymc()

    # Bundle is in INTERNAL units (solMass): 9.546e-4 solMass ≈ 1.0 jupiterMass
    internal_value = MJ_TO_MSUN
    posterior = {p_mass.label: np.array([internal_value])}

    # ACT — no param_lookup → no conversion
    result = p_double.generate_posterior(posterior, param_lookup=None)
    result_val = float(np.squeeze(result))

    # ASSERT: result should be 2× the internal input, still in solMass
    expected = 2.0 * internal_value
    assert np.isclose(result_val, expected, rtol=1e-3), (
        f"Expected {expected:.4e} solMass (internal); got {result_val:.4e}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4  –  get_draws
# ──────────────────────────────────────────────────────────────────────────────

def test_get_draws_converts_user_units_to_internal():
    """
    Given an idata whose posterior has planet.b.mass = 1.0 (user units, jupiterMass)
    and a param_lookup with the corresponding Parameter,
    When get_draws is called with param_lookup,
    Then each returned draw has planet.b.mass in internal units (~9.546e-4 solMass),
    ready for component physics functions that work in solMass.
    """
    # ARRANGE
    idata = _make_idata({"planet.b.mass": 1.0})  # 1.0 jupiterMass (user units)
    p = _mass_param()
    param_lookup = {p.label: p}

    # ACT
    draws = get_draws(idata, n_draws=4, param_lookup=param_lookup)

    # ASSERT — every draw should be in internal solMass
    for i, draw in enumerate(draws):
        val = float(np.atleast_1d(draw["planet.b.mass"])[0])
        assert np.isclose(val, MJ_TO_MSUN, rtol=1e-3), (
            f"Draw {i}: expected {MJ_TO_MSUN:.4e} solMass (internal); "
            f"got {val:.4e}. If ≈1.0, the user→internal conversion was not applied."
        )


def test_get_draws_without_param_lookup_returns_values_as_is():
    """
    Given the same idata, when get_draws is called without param_lookup,
    Then values are returned verbatim (the caller is responsible for units).
    """
    # ARRANGE
    user_value = 1.0
    idata = _make_idata({"planet.b.mass": user_value})

    # ACT
    draws = get_draws(idata, n_draws=4, param_lookup=None)

    # ASSERT
    for draw in draws:
        val = float(np.atleast_1d(draw["planet.b.mass"])[0])
        assert np.isclose(val, user_value, rtol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Regression: compute_summary on a scalar-float posterior
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_summary_handles_scalar_float_posterior():
    """
    Regression: generate_posterior returns a plain Python float (not a numpy
    array) for fixed/constant derived parameters (the no-inputs-in-posterior
    path).  When that float is stored in attr.posterior and compute_summary
    is called, it must not raise AttributeError on float.ndim.

    The crash appeared in to_latex_def() → compute_summary() after an end-to-
    end run where a derived parameter had no sampled dependencies.
    """
    # ARRANGE — simulate what _set_comp_posterior stores for a constant derived param
    p = _mass_param()
    p.posterior = 1.0   # plain float, as returned by generate_posterior's val.item()

    # ACT / ASSERT — must not raise AttributeError
    summary = p.compute_summary()

    assert np.isclose(summary.median, 1.0, rtol=1e-6), (
        f"Scalar-float posterior: expected median 1.0, got {summary.median}"
    )
