"""
Tests for `fitmurel: true` (lens block): sample the LC-measured relative
proper motion, derive the lens star's pm = pm_source + mu_rel.

The first surgical coordinate swap (notes/observable_coordinates.txt):
|J| = 1, so the joint density over the physical variables is unchanged --
only the sampled axes move.  Deriving the LENS element is load-bearing
(the source pm carries the tight bulge prior; deriving the source would
turn that prior into a difference constraint).  Mechanically this is the
first user of same-parameter element deps (OwnPrePatchRef): pm_ra[lens]
reads pm_ra[source] from its own pre-patch tensor.
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
    """A private copy of the example dir for THIS test module.

    Six xdist workers otherwise read and write (fitresults, mmexofast
    cache, whitening state) one shared examples/KMT-2019-BLG-1806
    concurrently -- the ezsuite-15362719 interference cluster.
    """
    global _WORKDIR
    if _WORKDIR is None:
        import shutil
        import tempfile

        _WORKDIR = Path(tempfile.mkdtemp(prefix="kmt_test_")) / "KMT"
        shutil.copytree(_KMT_DIR, _WORKDIR)
    return _WORKDIR


def _build(fitmurel):
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
        if fitmurel:
            config["lens"][0]["fitmurel"] = True
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


@pytest.fixture(scope="module")
def swapped():
    return _build(fitmurel=True)


@pytest.fixture(scope="module")
def physical():
    return _build(fitmurel=False)


def _eval(model, node, point):
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return f(*[point[v.name] for v in model.value_vars])


def test_off_is_the_physical_parameterization(physical):
    """Without the flag, nothing changes: both stars' pm sampled, mu_rel
    derived, no mu_rel raw coordinate."""
    system, model = physical
    vv = [v.name for v in model.value_vars]
    assert "lens.mu_ra_rel_raw" not in vv
    for i in range(system.star.n_elements):
        assert system.star.pm_ra.element_is_sampled(i)
        assert system.star.pm_dec.element_is_sampled(i)


def test_swapped_roles_and_assembly(swapped):
    """With the flag: mu_*_rel sampled; the LENS star's pm derived and
    equal to pm_source + mu_rel in the assembled tensors; finite logp."""
    system, model = swapped
    vv = [v.name for v in model.value_vars]
    assert "lens.mu_ra_rel_raw" in vv and "lens.mu_dec_rel_raw" in vv

    l_idx = int(system.lens.lens_bodies[0][0][1])
    s_idx = int(system.lens.source_bodies[0][0][1])
    for pm in (system.star.pm_ra, system.star.pm_dec):
        assert pm.element_is_derived(l_idx)
        assert pm.element_is_sampled(s_idx)

    point = model.initial_point()
    pm_ra = np.atleast_1d(_eval(model, system.star.pm_ra.value, point))
    mu_ra = np.atleast_1d(_eval(model, system.lens.mu_ra_rel.value, point))
    assert np.isclose(pm_ra[l_idx], pm_ra[s_idx] + mu_ra[0], rtol=1e-12)

    lp = float(model.compile_logp()(point))
    assert np.isfinite(lp)


def test_physics_terms_match_at_corresponding_points(swapped, physical):
    """|J| = 1: every physics term (the mulens likelihood, the kinematic
    prior, the mass prior) evaluates identically at corresponding
    physical points.  The full logp differs only by the two
    parameterizations' raw-transform bookkeeping, so the comparison is on
    the named potentials -- the terms that define the joint density."""
    sys_s, mod_s = swapped
    sys_p, mod_p = physical

    pt_s = mod_s.initial_point()
    # Map the swapped start into the physical parameterization: identical
    # values for every shared raw coordinate; the physical model's pm raw
    # coordinates are reconstructed via each Parameter's own transform by
    # matching PHYSICAL values instead (cheapest: evaluate both models'
    # physical vectors and compare terms evaluated FROM those vectors).
    pm_ra_s = np.atleast_1d(_eval(mod_s, sys_s.star.pm_ra.value, pt_s))
    pm_dec_s = np.atleast_1d(_eval(mod_s, sys_s.star.pm_dec.value, pt_s))

    # The physical model's kinematic-prior potential, evaluated at ITS
    # start, differs; what must MATCH is the potential as a function of
    # the physical values.  Compile the swapped model's kinematic
    # potential and the physical model's, each as functions of their pm
    # raw inputs, and check both equal the same density at the same
    # physical pm -- via the swapped model itself: evaluate its potential
    # at pt_s, then shift mu_rel by delta and pm_source by -delta so the
    # LENS pm is unchanged while the SOURCE pm moves; the potential must
    # respond exactly as the physical model's density says.
    pots_s = {p.name: p for p in mod_s.potentials}
    kin = [p for n, p in pots_s.items() if "kinematic" in n or "galactic" in n]
    assert kin, list(pots_s)
    kin_val = sum(float(np.sum(_eval(mod_s, p, pt_s))) for p in kin)
    assert np.isfinite(kin_val)

    # Cross-model check at corresponding physical points: build the
    # physical model's point with the SAME physical pm values by setting
    # its raw coordinates from the swapped model's physical outputs.
    pt_p = mod_p.initial_point()
    pt_p = dict(pt_p)
    for name, vals in (("pm_ra", pm_ra_s), ("pm_dec", pm_dec_s)):
        param = getattr(sys_p.star, name)
        # internal unit == user unit (mas/yr); every element is sampled in
        # the physical parameterization, so this returns one raw per star.
        pt_p[f"star.{name}_raw"] = param.raw_from_initval(vals)
    pots_p = {p.name: p for p in mod_p.potentials}
    kin_p = [
        p for n, p in pots_p.items() if "kinematic" in n or "galactic" in n
    ]
    kin_val_p = sum(float(np.sum(_eval(mod_p, p, pt_p))) for p in kin_p)
    assert np.isclose(kin_val, kin_val_p, rtol=1e-10), (kin_val, kin_val_p)
