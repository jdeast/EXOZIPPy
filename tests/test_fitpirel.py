"""
Tests for `fitpirel: true` (lens block): sample log_pi_rel, derive the
lens star's distance D_l = 1000/(pi_rel + 1000/D_s).

Swap 2 of the surgical coordinate plan.  Unlike fitmurel this map is
NONLINEAR, so a Jacobian potential (|dD_l/dlog_pi_rel|) accompanies it;
the finite-difference test below pins that the potential equals the
actual stretch of the map the model builds, not a formula transcribed
beside it.
"""

from pathlib import Path

import numpy as np
import pytensor
import pytest
import yaml

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


def _build(fitpirel):
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
        if fitpirel:
            config["lens"][0]["fitpirel"] = True
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


@pytest.fixture(scope="module")
def swapped():
    return _build(fitpirel=True)


def _eval(model, node, point):
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return f(*[point[v.name] for v in model.value_vars])


def test_off_is_the_physical_parameterization():
    system, model = _build(fitpirel=False)
    assert "lens.log_pi_rel_raw" not in [v.name for v in model.value_vars]
    for i in range(system.star.n_elements):
        assert system.star.distance.element_is_sampled(i)
    assert not any("fitpirel_jacobian" in p.name for p in model.potentials)


def test_swapped_roles_identity_and_jacobian(swapped):
    system, model = swapped
    assert "lens.log_pi_rel_raw" in [v.name for v in model.value_vars]

    l_idx = int(system.lens.lens_bodies[0][0][1])
    s_idx = int(system.lens.source_bodies[0][0][1])
    assert system.star.distance.element_is_derived(l_idx)
    assert system.star.distance.element_is_sampled(s_idx)

    point = model.initial_point()
    d = np.atleast_1d(_eval(model, system.star.distance.value, point))
    pr = np.atleast_1d(_eval(model, system.lens.pi_rel.value, point))
    assert np.isclose(
        d[l_idx], 1000.0 / (pr[0] + 1000.0 / d[s_idx]), rtol=1e-12
    )

    # The Jacobian potential equals the map's ACTUAL stretch: perturb the
    # raw log_pi_rel coordinate, measure dD_l/dlog_pi_rel by central
    # difference on the model graph itself, and compare to exp(potential).
    jac_pot = [p for p in model.potentials if "fitpirel_jacobian" in p.name]
    assert len(jac_pot) == 1
    jac_val = float(np.squeeze(_eval(model, jac_pot[0], point)))

    def at(delta):
        pt2 = dict(point)
        pt2["lens.log_pi_rel_raw"] = point["lens.log_pi_rel_raw"] + delta
        d2 = np.atleast_1d(_eval(model, system.star.distance.value, pt2))
        import math

        pr2 = np.atleast_1d(_eval(model, system.lens.pi_rel.value, pt2))
        return float(d2[l_idx]), math.log10(float(pr2[0]))

    eps = 1e-4
    d_hi, lp_hi = at(eps)
    d_lo, lp_lo = at(-eps)
    fd = abs((d_hi - d_lo) / (lp_hi - lp_lo))
    assert np.isclose(np.exp(jac_val), fd, rtol=1e-4), (np.exp(jac_val), fd)

    assert np.isfinite(float(model.compile_logp()(point)))
