"""
Tests for `fitthetae: true` (lens block): sample log_theta_E, derive the
HOST star's logmass from theta_E^2 = kappa * M_tot * pi_rel (with a
log_q companion, M_host = M_tot/(1+q)).

Swap 3 of the surgical coordinate plan.  In log coordinates the map is
LINEAR (logM = 2 log theta_E - log kappa pi_rel - log(1+q)), so unlike
fitpirel there is deliberately NO Jacobian potential -- |J| = 2 is
constant and cancels in the posterior.
"""

from pathlib import Path

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.constants import KAPPA
from exozippy.system import System

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"

_WORKDIR = None


def _kmt_workdir():
    global _WORKDIR
    if _WORKDIR is None:
        import shutil
        import tempfile

        _WORKDIR = Path(tempfile.mkdtemp(prefix="kmt_test_")) / "KMT"
        shutil.copytree(_KMT_DIR, _WORKDIR)
    return _WORKDIR


def _build(fitthetae, planet_linear=False, prepare_only=False):
    import os

    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")
    cwd = os.getcwd()
    os.chdir(_kmt_workdir())
    try:
        with open("KMT-2019-BLG-1806.yaml") as f:
            config = yaml.safe_load(f)
        with open(config["parameter_file"]) as f:
            user_params = yaml.safe_load(f)
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)
        if fitthetae:
            config["lens"][0]["fitthetae"] = True
        if planet_linear:
            config["planet"][0]["mass_parameterization"] = "linear"
        system = System(config, user_params=user_params)
        system.prepare()
        model = None if prepare_only else system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


def _eval(model, node, point):
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return f(*[point[v.name] for v in model.value_vars])


def test_off_is_the_physical_parameterization():
    system, _ = _build(fitthetae=False, prepare_only=True)
    assert "log_theta_E" not in system.lens.manifest
    entry = system.lens.manifest["theta_E"]
    assert entry.get("expr_key") == "default"


def test_swapped_roles_identity_no_jacobian():
    system, model = _build(fitthetae=True)
    assert "lens.log_theta_E_raw" in [v.name for v in model.value_vars]

    l_idx = int(system.lens.lens_bodies[0][0][1])
    assert system.star.logmass.element_is_derived(l_idx)

    point = model.initial_point()
    lm = np.atleast_1d(_eval(model, system.star.logmass.value, point))
    te = np.atleast_1d(_eval(model, system.lens.theta_E.value, point))
    pr = np.atleast_1d(_eval(model, system.lens.pi_rel.value, point))
    lq = np.atleast_1d(_eval(model, system.planet.log_q.value, point))
    expect = np.log10(te[0] ** 2 / (KAPPA * pr[0]) / (1.0 + 10.0 ** lq[0]))
    assert np.isclose(lm[l_idx], expect, rtol=1e-10)

    # Log-linear map: deliberately no correction potential.
    assert not any("thetae" in p.name.lower() for p in model.potentials)
    assert np.isfinite(float(model.compile_logp()(point)))


def test_linear_mass_companion_guard(caplog):
    """A companion sampling a linear mass has no log_q coordinate for the
    inverse; the flag warns and stands down in BOTH components."""
    import logging

    with caplog.at_level(logging.WARNING):
        system, _ = _build(
            fitthetae=True, planet_linear=True, prepare_only=True
        )
    assert "log_theta_E" not in system.lens.manifest
    assert any(
        "fitthetae" in r.message and "linear mass" in r.message
        for r in caplog.records
    )


def test_all_three_swaps_compose_into_observable_coordinates():
    """fitmurel + fitpirel + fitthetae + star_constrains_rho: false is the
    observable-coordinates parameterization of notes/
    observable_coordinates.txt: the sampled microlensing coordinates ARE
    the observable set, the whole lens-star physical state is derived,
    and every physical prior evaluates on the derived point."""
    import os

    cwd = os.getcwd()
    os.chdir(_kmt_workdir())
    try:
        with open("KMT-2019-BLG-1806.yaml") as f:
            config = yaml.safe_load(f)
        with open(config["parameter_file"]) as f:
            user_params = yaml.safe_load(f)
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)
        for flag in ("fitmurel", "fitpirel", "fitthetae"):
            config["lens"][0][flag] = True
        config["lens"][0]["star_constrains_rho"] = False
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)

    vv = [v.name for v in model.value_vars]
    for name in (
        "lens.mu_ra_rel_raw",
        "lens.mu_dec_rel_raw",
        "lens.log_pi_rel_raw",
        "lens.log_theta_E_raw",
        "lens.log_rho_raw",
    ):
        assert name in vv, name
    l_idx = int(system.lens.lens_bodies[0][0][1])
    for param in ("pm_ra", "pm_dec", "distance", "logmass"):
        assert getattr(system.star, param).element_is_derived(l_idx), param
    assert np.isfinite(float(model.compile_logp()(model.initial_point())))
