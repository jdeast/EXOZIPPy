"""
Tests for the per-lightcurve neighbor third light (``neighbor_flux``).

The parameter exists because the blend-tie assumption
``f_blend = SED-predicted flux of the modeled non-source stars`` is violated
as a rule at Roman resolution (unrelated line-of-sight stars dominate the
blend in most simulated events).  It is sampled ONLY on light curves with
``sed_constrains_blend: true``: with the tie off, f_blend is already free and
a neighbor term is exactly degenerate with it (the light curve measures only
the sum, finite-source or not).  With the tie on, positivity of the neighbor
flux converts the tie from the equality ``f_blend = f_lens_pred`` into the
physically correct inequality ``f_blend >= f_lens_pred``.

It is sampled as ``neighbor_fraction`` on [0, 1] and derived as
``neighbor_fraction * max(f_blend, 0)``, so its support is exactly the
physical ``[0, f_blend]`` with no flux scale in it (review 2.1.26(b)).
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

    Six xdist workers otherwise read and write (fitresults
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


def _load_kmt(constrain_blend_on=(), extra_params=None):
    """Load the KMT-2019-BLG-1806 example config + params, optionally
    switching ``sed_constrains_blend: true`` on the named light curves."""
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
        user_params.update(extra_params or {})
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)
        for entry in config["mulensinstrument"]:
            if entry["name"] in constrain_blend_on:
                entry["sed_constrains_blend"] = True
        system = System(config, user_params=user_params)
        system.prepare()
        return system
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def kmt_nb_system():
    """Given the KMT example with ``sed_constrains_blend: true`` on KMTC04
    only, when the system is prepared and built, provide
    (system, model, initial point)."""
    import os

    system = _load_kmt(constrain_blend_on=("KMTC04",))
    cwd = os.getcwd()
    os.chdir(_kmt_workdir())
    try:
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model, model.initial_point()


def _eval(model, node, point):
    """Evaluate a model node at ``point`` (see test_sed_flux_constraints)."""
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return f(*[point[v.name] for v in model.value_vars])


def test_no_gate_no_parameter():
    """
    Given the shipped KMT config (no sed_constrains_blend anywhere),
    When the system is prepared,
    Then neither neighbor_fraction nor neighbor_flux is declared -- the
    parameter only exists where the blend tie makes it identifiable.
    """
    system = _load_kmt()
    assert "neighbor_fraction" not in system.mulensinstrument.manifest
    assert "neighbor_flux" not in system.mulensinstrument.manifest


def test_gated_manifest_support_is_physical(kmt_nb_system):
    """
    Given sed_constrains_blend on KMTC04 only,
    When the model is built,
    Then neighbor_fraction element 0 is sampled on exactly [0, 1] -- the
    physical support, with no flux scale read from the data or the seed --
    from the middle of it, elements 1-2 are pinned at 0 (no third light),
    and neighbor_flux is DERIVED (review 2.1.26(b)).
    """
    system, model, point = kmt_nb_system
    inst = system.mulensinstrument
    frac = inst.neighbor_fraction
    i_on = list(inst.names).index("KMTC04")
    assert frac.element_is_sampled(i_on)
    assert float(frac.lower[i_on]) == 0.0
    assert float(frac.upper[i_on]) == 1.0
    assert float(frac.initval[i_on]) == 0.5
    for i in range(inst.n_elements):
        if i == i_on:
            continue
        assert not frac.element_is_sampled(i)
        assert frac.sigma[i] == 0.0
        assert frac.initval[i] == 0.0
    # Derived: no value variable of its own, only neighbor_fraction's.
    names = {v.name for v in model.value_vars}
    assert not any("neighbor_flux" in n for n in names)
    assert any("neighbor_fraction" in n for n in names)
    entry = inst.manifest["neighbor_fraction"]
    assert "upper" not in {k for k in entry if k != "overrides"}
    assert entry["overrides"]["upper"] == [1.0] * inst.n_elements
    assert entry["overrides"]["lower"] == [0.0] * inst.n_elements


def test_neighbor_flux_is_fraction_of_blend(kmt_nb_system):
    """
    Given the gated system at its start,
    When neighbor_flux is evaluated,
    Then it is neighbor_fraction * f_blend on the sampled element and 0 on
    the pinned ones -- so neighbor_flux <= f_blend holds exactly.
    """
    system, model, point = kmt_nb_system
    inst = system.mulensinstrument
    i_on = list(inst.names).index("KMTC04")
    f_b = np.atleast_1d(_eval(model, inst.f_blend.value, point))
    f_nb = np.atleast_1d(_eval(model, inst.neighbor_flux.value, point))
    frac = np.atleast_1d(_eval(model, inst.neighbor_fraction.value, point))
    assert f_b[i_on] > 0
    assert np.isclose(f_nb[i_on], frac[i_on] * f_b[i_on], rtol=1e-12)
    assert np.isclose(frac[i_on], 0.5, rtol=1e-12)
    for i in range(inst.n_elements):
        if i != i_on:
            assert f_nb[i] == 0.0


def test_calc_neighbor_flux_negative_blend_and_gradient():
    """
    Given a negative f_blend (q_source > 1: no room for third light),
    When calc_neighbor_flux and its gradient are evaluated,
    Then the flux is 0 and every gradient entry is finite -- the max()
    enters linearly, so its clamped side multiplies nothing infinite (the
    where-trap rule in CLAUDE.md).
    """
    import pytensor.tensor as pt

    from exozippy.components.mulensing.physics import calc_neighbor_flux

    frac = pt.dscalar("frac")
    fb = pt.dscalar("fb")
    out = calc_neighbor_flux(frac, fb)
    f_lens = 0.3
    logp = -0.5 * (2.5 * pt.log10((f_lens + out) / pt.maximum(fb, 1e-30))) ** 2
    fn = pytensor.function([frac, fb], [out, *pytensor.grad(logp, [frac, fb])])
    val, g_frac, g_fb = fn(0.7, -2.0)
    assert val == 0.0
    assert np.isfinite(g_frac) and np.isfinite(g_fb)
    val, g_frac, g_fb = fn(0.7, 2.0)
    assert np.isclose(val, 1.4)
    assert np.isfinite(g_frac) and np.isfinite(g_fb)


def test_blend_potential_matches_hand_formula(kmt_nb_system):
    """
    Given the gated system,
    When the sed_blend_prior potential is evaluated at the initial point,
    Then it equals the hand-computed flux-space residual
    -0.5 * (2.5*log10((f_lens_pred + f_nb) / f_blend) / sigma)**2,
    with f_lens_pred read off the SED prediction through the zeropoint --
    i.e. the old magnitude-difference potential with the neighbor flux
    added to the predicted blend (identical at f_nb = 0).
    """
    system, model, point = kmt_nb_system
    inst = system.mulensinstrument
    i_on = list(inst.names).index("KMTC04")

    pots = {p.name: p for p in model.potentials}
    name = f"{inst.prefix}.KMTC04.sed_blend_prior"
    assert name in pots
    got = float(_eval(model, pots[name], point))

    m_pred = float(
        _eval(
            model,
            system.sed.predict_blend_appmag(
                [
                    i
                    for i in range(system.star.n_elements)
                    if i not in inst._sed_source_indices(system)
                ],
                inst._sed_filter_keys(system)[i_on],
                system,
            ),
            point,
        )
    )
    zp = float(np.atleast_1d(_eval(model, inst.zeropoint.value, point))[i_on])
    f_b = float(np.atleast_1d(_eval(model, inst.f_blend.value, point))[i_on])
    f_nb = float(
        np.atleast_1d(_eval(model, inst.neighbor_flux.value, point))[i_on]
    )
    assert f_nb > 0  # the sampled start, not the pinned 0

    f_lens = 10 ** (-0.4 * (m_pred - zp))
    resid = 2.5 * np.log10(max(f_lens + f_nb, 1e-30) / max(f_b, 1e-30))
    expected = -0.5 * (resid / 0.2) ** 2
    assert np.isclose(got, expected, rtol=1e-10), (got, expected)


@pytest.fixture(scope="module")
def kmt_nb_user_system(kmt_nb_system):
    """Given sed_constrains_blend on KMTC04 and KMTS04, and a params file
    that TIGHTENS KMTC04's neighbor_fraction upper (0.5) and sets its start,
    and tries to LOOSEN KMTS04's upper (2.0, past the physical 1), provide
    the prepared and built system."""
    import os

    system = _load_kmt(
        constrain_blend_on=("KMTC04", "KMTS04"),
        extra_params={
            "mulensinstrument.KMTC04.neighbor_fraction": {
                "upper": 0.5,
                "initval": 0.3,
            },
            "mulensinstrument.KMTS04.neighbor_fraction": {"upper": 2.0},
        },
    )
    cwd = os.getcwd()
    os.chdir(_kmt_workdir())
    try:
        system.build_model()
    finally:
        os.chdir(cwd)
    return system


def test_user_may_tighten_but_not_loosen(kmt_nb_user_system):
    """
    Given the user bounds above,
    When the model is built,
    Then the tighter upper stands, the looser one is clipped back to the
    physical 1 (the validity-limit channel; resolve warns, naming it), and
    the element the user did not bound keeps 1.
    """
    system = kmt_nb_user_system
    names = list(system.mulensinstrument.names)
    assert names[:2] == ["KMTC04", "KMTS04"]
    upper = np.asarray(
        system.mulensinstrument.neighbor_fraction.upper, dtype=float
    )
    assert upper[0] == 0.5
    assert upper[1] == 1.0
    assert upper[2] == 1.0


def test_loosening_warns(caplog):
    """
    Given a params-file neighbor_fraction upper past 1,
    When the system is prepared,
    Then resolve warns that the user bound was not applied (the stricter,
    physical bound wins).
    """
    import logging

    with caplog.at_level(logging.WARNING):
        _load_kmt(
            constrain_blend_on=("KMTC04",),
            extra_params={
                "mulensinstrument.KMTC04.neighbor_fraction": {"upper": 2.0}
            },
        )
    msgs = [
        r.getMessage()
        for r in caplog.records
        if "neighbor_fraction" in r.getMessage()
        and "user upper=2" in r.getMessage()
    ]
    assert msgs, [r.getMessage() for r in caplog.records][-20:]


def test_user_start_beats_the_default(kmt_nb_user_system):
    """
    Given a user initval on KMTC04 and none on KMTS04,
    When the model is built,
    Then KMTC04 starts at the user's value and KMTS04 at the 0.5 default.
    """
    init = np.asarray(
        kmt_nb_user_system.mulensinstrument.neighbor_fraction.initval,
        dtype=float,
    )
    assert np.isclose(init[0], 0.3, rtol=1e-12, atol=0.0)
    assert np.isclose(init[1], 0.5, rtol=1e-12, atol=0.0)
