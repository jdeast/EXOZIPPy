"""
Tests for `fitu0te: true` (lens block): sample the SIGNED effective
timescale u0te = u_0 * t_E (days), derive u_0 = u0te / t_E.

Swap 4 of the surgical coordinate plan.  Measured motivation (event 128,
murel arm): corr(log u_0, log t_E) = -0.96 and t_eff is 3x tighter than
u_0.  Signed and linear because u_0 itself is signed (the +/- reflection
is a sampled mode); |du_0/du0te| = 1/t_E is not constant, so a Jacobian
potential accompanies the swap and is pinned here by finite difference
against the model's own map.  The config name is u0te, NOT t_eff: the
stellar effective temperature owns 'teff' in the config namespace
(star.Lens.teff vs lens.Lens.teff differ only in the component).
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


def _build(fitu0te):
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
        if fitu0te:
            config["lens"][0]["fitu0te"] = True
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


def _eval(model, node, point):
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return f(*[point[v.name] for v in model.value_vars])


def test_off_is_the_physical_parameterization():
    system, model = _build(fitu0te=False)
    vv = [v.name for v in model.value_vars]
    assert "lens.u0te_raw" not in vv
    assert "lens.u_0_raw" in vv
    assert not any("fitu0te" in p.name for p in model.potentials)


def test_swapped_identity_and_fd_jacobian():
    system, model = _build(fitu0te=True)
    vv = [v.name for v in model.value_vars]
    assert "lens.u0te_raw" in vv and "lens.u_0_raw" not in vv

    point = model.initial_point()
    u0 = np.atleast_1d(_eval(model, system.lens.u_0.value, point))
    ut = np.atleast_1d(_eval(model, system.lens.u0te.value, point))
    te = np.atleast_1d(_eval(model, system.lens.t_E.value, point))
    assert np.isclose(u0[0], ut[0] / te[0], rtol=1e-12)

    jac_pot = [p for p in model.potentials if "fitu0te_jacobian" in p.name]
    assert len(jac_pot) == 1
    jac_val = float(np.squeeze(_eval(model, jac_pot[0], point)))

    def at(delta):
        pt2 = dict(point)
        pt2["lens.u0te_raw"] = point["lens.u0te_raw"] + delta
        u = float(np.atleast_1d(_eval(model, system.lens.u_0.value, pt2))[0])
        w = float(np.atleast_1d(_eval(model, system.lens.u0te.value, pt2))[0])
        return u, w

    eps = 1e-4
    u_hi, w_hi = at(eps)
    u_lo, w_lo = at(-eps)
    fd = abs((u_hi - u_lo) / (w_hi - w_lo))
    assert np.isclose(np.exp(jac_val), fd, rtol=1e-6), (np.exp(jac_val), fd)

    assert np.isfinite(float(model.compile_logp()(point)))
