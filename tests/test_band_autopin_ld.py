"""Band pins its own limb darkening when nothing in the topology reads it.

A `band:` block declares filter identity.  For a point-source microlensing
fit that is the whole reason to have one -- the magnification model only takes
u1 when `finite_source: true` -- so adding a band used to add two free RVs
(band.q1_raw/q2_raw) that no likelihood term touched, and moved the start logp
by several nats.  Band now pins the LD parameters of every band no consumer
reads, through the manifest "overrides" channel (layered UNDER the params
file, so an explicit user entry still frees it).

The consumers, and the condition under which each reads a band's LD:
  transit           -- unconditional (any transit referencing the band)
  mulensinstrument  -- only when lens.finite_source is on
  rvinstrument rm:  -- the `rm_band` band, or band 0 when unset
  astrometry band:  -- NOT a consumer (filter identity only, for the SED)
"""

import logging

import numpy as np
import pytest

from exozippy.system import System

# --------------------------------------------------------------------------
# Fixtures: a minimal point-source PSPL fit, with and without a band block.
# --------------------------------------------------------------------------
T0 = 2460025.0
TE = 30.0
U0 = 0.1


def _write_pspl_lc(path, n=60):
    """Synthetic noiseless PSPL light curve in magnitudes."""
    t = np.linspace(T0 - 2 * TE, T0 + 2 * TE, n)
    u = np.sqrt(U0**2 + ((t - T0) / TE) ** 2)
    amp = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    mag = 18.0 - 2.5 * np.log10(amp)
    err = np.full(n, 0.01)
    np.savetxt(path, np.column_stack([t, mag, err]))
    return str(path)


def _mulens_config(lc, bands=None, mulens_band="I", finite_source=False):
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [
            {
                "name": "Lens",
                "lens_ndx": 0,
                "source_ndx": 1,
                "finite_source": finite_source,
                "t0_par": T0,
                "use_op": False,
                # Never shell out to MMEXOFAST from a unit test.
                "mmexofast": False,
            }
        ],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }
    if bands is not None:
        config["band"] = bands
        if mulens_band is not None:
            config["mulensinstrument"][0]["band"] = mulens_band
    return config


def _mulens_params(finite_source=False):
    params = {
        "lens.Lens.t_0": {"initval": T0},
        "lens.Lens.u_0": {"initval": U0},
        "lens.Lens.t_E": {"initval": TE},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("Lens", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    if finite_source:
        params["lens.Lens.rho"] = {"initval": 1.0e-3}
    return params


def _band_rv_names(model):
    return sorted(v.name for v in model.free_RVs if v.name.startswith("band."))


def _build(config, params):
    system = System(config, user_params=params)
    system.prepare()
    return system, system.build_model()


@pytest.fixture(scope="module")
def pspl_lc(tmp_path_factory):
    return _write_pspl_lc(tmp_path_factory.mktemp("band_autopin") / "lc.dat")


@pytest.fixture(scope="module")
def point_source_with_band(pspl_lc):
    """Given a point-source PSPL fit whose light curve declares a band,
    when the system is prepared and built, provide (system, model)."""
    return _build(
        _mulens_config(pspl_lc, bands=[{"name": "I", "filter": "I"}]),
        _mulens_params(),
    )


# --------------------------------------------------------------------------
# 1. Point-source microlensing: the band's LD is unread, so the parameters
#    are omitted from the manifest entirely (no table rows at all).
# --------------------------------------------------------------------------
def test_point_source_mulens_band_has_no_free_ld_rvs(point_source_with_band):
    """
    Given a point-source microlensing fit with a band: block,
    When the model is built,
    Then no band limb-darkening parameter is a free RV.
    """
    _system, model = point_source_with_band
    assert _band_rv_names(model) == []


def test_unread_band_has_no_ld_parameters_at_all(point_source_with_band):
    """
    Given the same fit,
    When the Band component is inspected,
    Then no limb-darkening parameter exists -- not even pinned: an
    unconsumed band contributes nothing to the manifest, so the table
    carries no Band section for it.
    """
    system, _model = point_source_with_band
    assert "q1" not in system.band.manifest
    assert "q2" not in system.band.manifest
    assert not hasattr(system.band, "q1")
    assert not hasattr(system.band, "q2")


def test_band_block_costs_nothing_when_its_ld_is_unread(pspl_lc):
    """
    Given the same fit with and without the band: block,
    When each model is built,
    Then the two have the same free-RV set and the same start logp, bit for
    bit -- declaring filter identity is free.
    """
    system_b, model_b = _build(
        _mulens_config(pspl_lc, bands=[{"name": "I", "filter": "I"}]),
        _mulens_params(),
    )
    system_n, model_n = _build(_mulens_config(pspl_lc), _mulens_params())

    names_b = sorted(v.name for v in model_b.free_RVs)
    names_n = sorted(v.name for v in model_n.free_RVs)
    assert names_b == names_n

    lp_b = float(model_b.compile_logp()(system_b.get_raw_start(model_b)))
    lp_n = float(model_n.compile_logp()(system_n.get_raw_start(model_n)))
    assert lp_b == lp_n


def test_omission_is_logged_at_info(pspl_lc, caplog):
    """
    Given a band whose limb darkening nothing reads,
    When the system is prepared,
    Then an INFO record says the LD parameters were omitted and why.
    """
    with caplog.at_level(logging.INFO, logger="exozippy.components.band.band"):
        system = System(
            _mulens_config(pspl_lc, bands=[{"name": "I", "filter": "I"}]),
            user_params=_mulens_params(),
        )
        system.prepare()

    msgs = [
        r.message
        for r in caplog.records
        if "no limb-darkening parameters" in r.message
    ]
    assert len(msgs) == 1, msgs
    assert "nothing in this topology reads" in msgs[0]


# --------------------------------------------------------------------------
# 2. Flipping finite_source is the only edit needed to free the LD again.
# --------------------------------------------------------------------------
def test_finite_source_frees_the_limb_darkening(pspl_lc):
    """
    Given the same fit with lens.finite_source: true and nothing else changed,
    When the model is built,
    Then the band's q1/q2 are free RVs again.
    """
    _system, model = _build(
        _mulens_config(
            pspl_lc,
            bands=[{"name": "I", "filter": "I"}],
            finite_source=True,
        ),
        _mulens_params(finite_source=True),
    )
    assert _band_rv_names(model) == ["band.q1_raw", "band.q2_raw"]


# --------------------------------------------------------------------------
# 3. With NO consumer the parameters do not exist, so a params entry cannot
#    resurrect them (the pin-under-override escape applies only to the
#    mixed case, where some other band forces the vector to exist).
# --------------------------------------------------------------------------
def test_user_entry_cannot_resurrect_an_unconsumed_ld(pspl_lc):
    """
    Given a point-source fit whose params file gives band.I.q1 a sigma,
    When the model is built,
    Then q1 still does not exist: nothing in the topology could consume
    it, so a free q1 would be a parameter with no likelihood term.
    """
    params = _mulens_params()
    params["band.I.q1"] = {"initval": 0.4, "sigma": 0.1}
    system, model = _build(
        _mulens_config(pspl_lc, bands=[{"name": "I", "filter": "I"}]), params
    )
    assert _band_rv_names(model) == []
    assert not hasattr(system.band, "q1")


# --------------------------------------------------------------------------
# 4. A linear-law band with no consumer omits u1 too, and still builds.
# --------------------------------------------------------------------------
def test_linear_law_band_omits_u1_and_still_builds(pspl_lc):
    """
    Given a point-source fit whose band declares ld_law: linear,
    When the model is built,
    Then u1 does not exist, and the build order resolves.
    """
    system, model = _build(
        _mulens_config(
            pspl_lc,
            bands=[{"name": "I", "filter": "I", "ld_law": "linear"}],
        ),
        _mulens_params(),
    )
    assert _band_rv_names(model) == []
    assert not hasattr(system.band, "u1")


# --------------------------------------------------------------------------
# 5. Transit reads limb darkening unconditionally; the predicate is per band.
# --------------------------------------------------------------------------
def _write_transit_lc(path, t0=2459634.3, n=120):
    rng = np.random.default_rng(42)
    t = np.linspace(t0 - 0.2, t0 + 0.2, n)
    flux = 1.0 + rng.normal(0.0, 1e-3, n)
    np.savetxt(path, np.column_stack([t, flux, np.full(n, 1e-3)]))
    return str(path)


def _transit_params():
    return {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207, "sigma": 100},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": 2.99},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": 1.7},
    }


def test_transit_band_ld_stays_free_and_an_unused_band_is_pinned(
    tmp_path_factory,
):
    """
    Given two bands where only the first is referenced by a transit,
    When the model is built,
    Then the transit's band keeps free q1/q2 and the other band is pinned --
    the predicate is per band instance, not per system.
    """
    d = tmp_path_factory.mktemp("band_autopin_transit")
    lc = _write_transit_lc(d / "lc.dat")
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [
            {"name": "TESS", "filter": "TESS"},
            {"name": "V", "filter": "V"},
        ],
        "transit": [{"name": "inst0", "file": lc, "band": "TESS"}],
    }
    system, model = _build(config, _transit_params())

    assert _band_rv_names(model) == ["band.q1_raw", "band.q2_raw"]
    # Element 0 (TESS) free, element 1 (V) pinned.
    for param in (system.band.q1, system.band.q2):
        sigma = np.atleast_1d(param.sigma)
        assert np.isnan(sigma[0]) or sigma[0] != 0.0
        assert sigma[1] == 0.0


def test_two_transits_may_use_different_limb_darkening_laws(
    tmp_path_factory,
):
    """
    Given two transits in two bands, one declaring the quadratic law and the
      other the linear one,
    When the model is built,
    Then the quadratic band samples the Kipping pair while the linear band
      samples u1 itself, u1 is derived on the quadratic band only, u2 is
      exactly 0 on the linear band, and the start's logp and gradient are
      finite.

    This configuration used to RAISE ("all bands must use the same ld_law"),
    because Parameter.build_pymc derived a whole vector or none of it; the
    documented workaround was quadratic everywhere with the linear band's q2
    pinned at 0.5, which reproduces u2 = 0 but samples uniformly in q1 rather
    than in u1.  The per-element roles express it directly.
    """
    d = tmp_path_factory.mktemp("band_mixed_laws")
    lc0 = _write_transit_lc(d / "lc0.dat")
    lc1 = _write_transit_lc(d / "lc1.dat")
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [
            {"name": "TESS", "filter": "TESS", "ld_law": "quadratic"},
            {"name": "V", "filter": "V", "ld_law": "linear"},
        ],
        "transit": [
            {"name": "inst0", "file": lc0, "band": "TESS"},
            {"name": "inst1", "file": lc1, "band": "V"},
        ],
    }
    system, model = _build(config, _transit_params())

    # Both coordinate sets are sampled, each on its own band.
    assert _band_rv_names(model) == [
        "band.q1_raw",
        "band.q2_raw",
        "band.u1_raw",
    ]
    assert system.band.q1.is_sampled.tolist() == [True, False]
    assert system.band.u1.is_sampled.tolist() == [False, True]
    assert system.band.u1.is_derived.tolist() == [True, False]
    # The Kipping coordinates and u2 are not parameters of the linear band.
    assert system.band.q1.is_active.tolist() == [True, False]
    assert system.band.u2.is_active.tolist() == [True, False]

    with model:
        point = model.initial_point()
        u2 = model.compile_fn(
            model.replace_rvs_by_values([system.band.u2.value]),
            inputs=model.value_vars,
            point_fn=True,
            on_unused_input="ignore",
        )(point)[0]
        logp = model.compile_logp()(point)
        dlogp = model.compile_dlogp()(point)

    assert float(np.atleast_1d(u2)[1]) == 0.0
    assert np.isfinite(logp)
    assert np.all(np.isfinite(dlogp))

    # The roles reach a trace, so mkparam -- which has no System -- writes a
    # start value for the LINEAR band's u1 and none for the quadratic band's
    # (whose value is an expression).  Checked here rather than in a test of
    # its own because the model this needs is already built.
    from exozippy.trace_meta import element_roles

    roles = element_roles(system)
    assert roles["band.u1"]["sampled"] == [False, True]
    assert roles["band.u1"]["derived"] == [True, False]
    assert roles["band.u2"]["active"] == [True, False]


# --------------------------------------------------------------------------
# 6. The predicate itself, on stub topologies: the cases a full System build
#    would be far too expensive to cover (RM, astrometry).
# --------------------------------------------------------------------------
class _StubComp:
    def __init__(self, config):
        self.config = config


class _StubSystem:
    def __init__(self, **comps):
        for name, cfg in comps.items():
            setattr(self, name, _StubComp(cfg))


def _band_for(names):
    from conftest import _DummyConfigManager
    from exozippy.components.band.band import Band

    band = Band(
        [{"name": n, "filter": n} for n in names], _DummyConfigManager()
    )
    band.load_data(system=None)
    return band


@pytest.mark.parametrize(
    "rv_config, expected",
    [
        # `rm:` with no `rm_band:` reads band 0 (rm.resolve_rm_indices).
        ([{"name": "TRES", "rm": "b"}], {0}),
        # `rm_band:` names the band explicitly.
        ([{"name": "TRES", "rm": "b", "rm_band": "V"}], {1}),
        # No `rm:` key -> the RV model never touches limb darkening.
        ([{"name": "TRES", "rm_band": "V"}], set()),
    ],
)
def test_rm_consumer_indices(rv_config, expected):
    """
    Given an rvinstrument with (or without) a Rossiter-McLaughlin request,
    When the LD consumer set is computed,
    Then only the band the RM model actually reads is a consumer.
    """
    band = _band_for(["TESS", "V"])
    system = _StubSystem(rvinstrument=rv_config)
    assert set(band._ld_consumer_indices(system)) == expected


def test_astrometry_band_is_not_an_ld_consumer():
    """
    Given an astrometryinstrument referencing a band for its SED fluxfrac,
    When the LD consumer set is computed,
    Then the band is not a consumer -- astrometry uses filter identity only.
    """
    band = _band_for(["V"])
    system = _StubSystem(
        astrometryinstrument=[{"name": "gaia", "band": "V", "mode": "gaia"}]
    )
    assert set(band._ld_consumer_indices(system)) == set()


def test_point_source_mulens_band_is_not_an_ld_consumer():
    """
    Given microlensing light curves on a band but a point-source lens,
    When the LD consumer set is computed,
    Then the band is not a consumer; turning finite_source on makes it one.
    """
    band = _band_for(["I"])
    point = _StubSystem(
        lens=[{"finite_source": False}],
        mulensinstrument=[{"name": "OGLE", "band": "I"}],
    )
    finite = _StubSystem(
        lens=[{"finite_source": True}],
        mulensinstrument=[{"name": "OGLE", "band": "I"}],
    )
    assert set(band._ld_consumer_indices(point)) == set()
    assert set(band._ld_consumer_indices(finite)) == {0}
