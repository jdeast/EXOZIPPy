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
    Then neighbor_flux is not declared at all -- the parameter only exists
    where the blend tie makes it identifiable.
    """
    system = _load_kmt()
    assert "neighbor_flux" not in system.mulensinstrument.manifest


def test_gated_manifest_pins_and_scales(kmt_nb_system):
    """
    Given sed_constrains_blend on KMTC04 only,
    When the model is built,
    Then neighbor_flux element 0 is sampled and elements 1-2 are pinned at
    the defaults.yaml 0.0, and upper/initval scale with each light curve's
    MEDIAN OBSERVED flux -- a property of the data, not of the bootstrap's
    seed-dependent baseline (fs_init), so the same light curve gets the same
    prior support whatever the seed or the model topology.
    """
    system, model, point = kmt_nb_system
    inst = system.mulensinstrument
    nb = inst.neighbor_flux
    scale = np.asarray(inst.flux_median_init, dtype=float)
    # Non-vacuity: the median and the bootstrap genuinely differ here, so
    # the assertions below say which one the bound reads.
    assert not np.allclose(
        scale, np.asarray(inst.fs_init, dtype=float), rtol=1e-6, atol=0.0
    )

    i_on = list(inst.names).index("KMTC04")
    assert nb.element_is_sampled(i_on)
    assert np.isclose(
        nb.initval[i_on], 0.05 * scale[i_on], rtol=1e-12, atol=0.0
    )
    for i in range(inst.n_elements):
        if i == i_on:
            continue
        assert not nb.element_is_sampled(i)
        assert nb.sigma[i] == 0.0
        assert nb.initval[i] == 0.0  # no hint: the defaults.yaml 0.0
    assert np.allclose(
        np.asarray(nb.upper, dtype=float), 2.0 * scale, rtol=1e-12, atol=0.0
    )


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
    assert f_nb > 0  # the scaled initval, not the pinned default

    f_lens = 10 ** (-0.4 * (m_pred - zp))
    resid = 2.5 * np.log10(max(f_lens + f_nb, 1e-30) / max(f_b, 1e-30))
    expected = -0.5 * (resid / 0.2) ** 2
    assert np.isclose(got, expected, rtol=1e-10), (got, expected)

    # One-sidedness is then algebra on the matched formula: a blend brighter
    # than the lens prediction is absorbed exactly by f_nb = f_b - f_lens >= 0,
    # while a blend fainter than the lens prediction cannot be (f_nb >= 0
    # keeps resid >= 2.5*log10(f_lens/f_b) > 0).  Pin the support here so a
    # bounds edit cannot silently break that argument.
    assert float(np.atleast_1d(inst.neighbor_flux.lower)[i_on]) == 0.0


def test_no_user_value_writes_nothing_and_ranks_the_start(kmt_nb_system):
    """
    Given the gated system and no user neighbor_flux entry,
    When it is prepared,
    Then the start is a ranked DERIVED_DATA hint (provenance, not an
    "overrides" initval), nothing reached user_params, and the upper is a
    manifest OPTION, not an override (review 2.1.26(b)).
    """
    from exozippy.config import PRECEDENCE_DERIVED_DATA

    system, _, _ = kmt_nb_system
    inst = system.mulensinstrument
    cm = system.config_manager
    i_on = list(inst.names).index("KMTC04")
    path = f"{inst.prefix}.{i_on}.neighbor_flux"
    assert cm.hint_ranks[path] == PRECEDENCE_DERIVED_DATA
    # The engine's own inject-back of a solved start ({initval, derived})
    # is expected; a bound in user_params would be a component write.
    assert not any(
        "upper" in v for k, v in cm.user_params.items() if "neighbor_flux" in k
    )
    entry = inst.manifest["neighbor_flux"]
    assert "upper" not in (entry.get("overrides") or {})
    assert "initval" not in (entry.get("overrides") or {})
    assert len(entry["upper"]) == inst.n_elements


@pytest.fixture(scope="module")
def kmt_nb_user_system(kmt_nb_system):
    """Given sed_constrains_blend on KMTC04 and KMTS04, and a params file
    that LOOSENS KMTC04's upper (10x the median flux, past the 2x default)
    and sets its start, and TIGHTENS KMTS04's upper (0.5x), provide
    (system, median-flux scale)."""
    import os

    scale = np.asarray(
        kmt_nb_system[0].mulensinstrument.flux_median_init, dtype=float
    )
    system = _load_kmt(
        constrain_blend_on=("KMTC04", "KMTS04"),
        extra_params={
            "mulensinstrument.KMTC04.neighbor_flux": {
                "upper": float(10.0 * scale[0]),
                "initval": float(0.3 * scale[0]),
            },
            "mulensinstrument.KMTS04.neighbor_flux": {
                "upper": float(0.5 * scale[1]),
            },
        },
    )
    cwd = os.getcwd()
    os.chdir(_kmt_workdir())
    try:
        system.build_model()
    finally:
        os.chdir(cwd)
    return system, scale


def test_user_may_loosen_and_tighten_the_upper(kmt_nb_user_system):
    """
    Given the user bounds above,
    When the model is built,
    Then each user upper stands as written -- looser AND tighter than the
    2x default -- while the element the user did not bound keeps 2x.  Under
    the old "overrides" upper the loosened bound was clipped back to 2x.
    """
    system, scale = kmt_nb_user_system
    names = list(system.mulensinstrument.names)
    upper = np.asarray(
        system.mulensinstrument.neighbor_flux.upper, dtype=float
    )
    assert names[:2] == ["KMTC04", "KMTS04"]
    assert np.isclose(upper[0], 10.0 * scale[0], rtol=1e-12, atol=0.0)
    assert np.isclose(upper[1], 0.5 * scale[1], rtol=1e-12, atol=0.0)
    assert np.isclose(upper[2], 2.0 * scale[2], rtol=1e-12, atol=0.0)


def test_user_start_beats_the_hint(kmt_nb_user_system):
    """
    Given a user initval on KMTC04 and none on KMTS04,
    When the model is built,
    Then KMTC04 starts at the user's value and KMTS04 at the 0.05x hint.
    """
    system, scale = kmt_nb_user_system
    init = np.asarray(
        system.mulensinstrument.neighbor_flux.initval, dtype=float
    )
    assert np.isclose(init[0], 0.3 * scale[0], rtol=1e-12, atol=0.0)
    assert np.isclose(init[1], 0.05 * scale[1], rtol=1e-12, atol=0.0)
