"""
Tests for `fitu0te: true` (a per-SOURCE flag, carried on each `source:`
entry post-split): sample the SIGNED effective timescale
u0te = u_0 * t_E (days), derive u_0 = u0te / t_E.

Swap 4 of the surgical coordinate plan.  Measured motivation (event 128,
murel arm): corr(log u_0, log t_E) = -0.96 and t_eff is 3x tighter than
u_0.  Signed and linear because u_0 itself is signed (the +/- reflection
is a sampled mode); |du_0/du0te| = 1/t_E is not constant, so a Jacobian
potential accompanies the swap and is pinned here by finite difference
against the model's own map.  The config name is u0te, NOT t_eff: the
stellar effective temperature owns 'teff' in the config namespace
(star.Source.teff vs source.Source.teff differ only in the component).
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
            config["source"][0]["fitu0te"] = True
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
    assert "source.u0te_raw" not in vv
    assert "source.u_0_raw" in vv
    assert not any("fitu0te" in p.name for p in model.potentials)


def test_swapped_identity_and_fd_jacobian():
    system, model = _build(fitu0te=True)
    vv = [v.name for v in model.value_vars]
    assert "source.u0te_raw" in vv and "source.u_0_raw" not in vv

    point = model.initial_point()
    u0 = np.atleast_1d(_eval(model, system.source.u_0.value, point))
    ut = np.atleast_1d(_eval(model, system.source.u0te.value, point))
    te = np.atleast_1d(_eval(model, system.mulensevent.t_E.value, point))
    assert np.isclose(u0[0], ut[0] / te[0], rtol=1e-12)

    jac_pot = [p for p in model.potentials if "fitu0te_jacobian" in p.name]
    assert len(jac_pot) == 1
    jac_val = float(np.squeeze(_eval(model, jac_pot[0], point)))

    def at(delta):
        pt2 = dict(point)
        pt2["source.u0te_raw"] = point["source.u0te_raw"] + delta
        u = float(np.atleast_1d(_eval(model, system.source.u_0.value, pt2))[0])
        w = float(
            np.atleast_1d(_eval(model, system.source.u0te.value, pt2))[0]
        )
        return u, w

    eps = 1e-4
    u_hi, w_hi = at(eps)
    u_lo, w_lo = at(-eps)
    fd = abs((u_hi - u_lo) / (w_hi - w_lo))
    assert np.isclose(np.exp(jac_val), fd, rtol=1e-6), (np.exp(jac_val), fd)

    assert np.isfinite(float(model.compile_logp()(point)))


# ---------------------------------------------------------------------------
# Review 1.6.8: the Jacobian on a MULTI-source event.  Pre-split the
# potential covered source slot 0 only (-log t_E[0]) while the swap applied
# per source, so a 2-source fit was misweighted by a factor t_E.  Post-split
# t_E is the EVENT's one scalar and Source.build_likelihood applies
# -n_u0te * log(t_E): one -log t_E per OPTED-IN track.  Pinned here by the
# finite-difference determinant of the (u0te_j -> u_0_j) map, for both
# sources opted in and for a mixed pair.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["both", "mixed"])
def two_source_fitu0te(request, tmp_path_factory):
    from mulens_synthetic import (
        build,
        mulens_config,
        mulens_params,
        write_flat_lc,
    )

    lc = write_flat_lc(tmp_path_factory.mktemp("u0te2s") / "lc.dat")
    second = {"fitu0te": True} if request.param == "both" else {}
    config = mulens_config(
        lc, sources=(("S1", {"fitu0te": True}), ("S2", second))
    )
    system, model = build(config, mulens_params(config))
    return request.param, system, model


@pytest.mark.slow
def test_two_source_fd_jacobian(two_source_fitu0te):
    """
    Given: a 2-source event with fitu0te on BOTH sources, or on source 0
      only,
    When: the model is built,
    Then: exp(fitu0te_jacobian) equals the finite-difference |det| of the
      map from the OPTED-IN u0te elements to their u_0 elements -- 1/t_E
      per opted-in source, (1/t_E)**2 with both.  The pre-split single-slot
      term gave 1/t_E for "both", off by a factor t_E.
    """
    kind, system, model = two_source_fitu0te
    opted = [0, 1] if kind == "both" else [0]
    point = model.initial_point()

    jac_pot = [p for p in model.potentials if "fitu0te_jacobian" in p.name]
    assert len(jac_pot) == 1
    jac_val = float(np.squeeze(_eval(model, jac_pot[0], point)))

    def at(j, delta):
        pt2 = dict(point)
        raw = np.array(point["source.u0te_raw"], dtype=float).copy()
        raw[j] += delta
        pt2["source.u0te_raw"] = raw
        u = np.atleast_1d(_eval(model, system.source.u_0.value, pt2))
        w = np.atleast_1d(_eval(model, system.source.u0te.value, pt2))
        return u[opted], w[opted]

    eps = 1e-4
    jac = np.zeros((len(opted), len(opted)))
    for col, j in enumerate(opted):
        u_hi, w_hi = at(j, eps)
        u_lo, w_lo = at(j, -eps)
        jac[:, col] = (u_hi - u_lo) / (w_hi[col] - w_lo[col])
    fd = abs(np.linalg.det(jac))
    assert np.isclose(np.exp(jac_val), fd, rtol=1e-6), (
        kind,
        np.exp(jac_val),
        fd,
    )

    t_E = np.atleast_1d(_eval(model, system.mulensevent.t_E.value, point))
    assert t_E.size == 1  # ONE event-level t_E shared by both tracks
    assert np.isclose(jac_val, -len(opted) * np.log(t_E[0]), rtol=1e-12)
    assert np.isfinite(float(model.compile_logp()(point)))
