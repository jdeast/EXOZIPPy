"""Exposure smearing of astrometry (instrument.md, "Exposure smearing").

An astrometric exposure measures the time-mean PHOTOCENTER POSITION, so a
dataset with ``ninterp > 1`` averages the sky vector (dE, dN) over each
exposure -- both components together, on one set of Gauss-Legendre nodes --
and forms its observables (the along-scan abscissa, the separation and the
position angle) from the averaged vector afterwards:

  - rel: the smeared (dE, dN) is the numerically averaged track, and the
    separation and PA come from it (not from averaging sep and PA, which
    differ on a curving arc and are undefined across the PA wrap);
  - gaia: the along-scan projection of the smeared vector equals the
    smeared along-scan abscissa (the projection is linear);
  - abs: the smeared vector is the averaged track, linear terms included;
  - the plotted model is the likelihood's own (the builder contract);
  - ninterp 1 is the instantaneous model exactly, and a catalog dataset
    refuses the keys.

The orbit is fast (P = 1 d) and the exposures long (6 h, a quarter orbit)
so the average differs from the mid-exposure value by far more than any
tolerance used here.
"""

import numpy as np
import pytest

from exozippy.system import System

P_D = 1.0
EXPTIME_S = 21600.0  # 6 h = P/4
NINTERP = 5
TC = 2457000.3
EPOCH = 2457400.0


def _write(tmp_dir):
    rng = np.random.default_rng(3)
    t_g = np.sort(rng.uniform(2456900.0, 2457900.0, 30))
    psi = rng.uniform(0.0, 360.0, 30)
    np.savetxt(
        tmp_dir / "g.astrom",
        np.column_stack([t_g, np.zeros(30), np.full(30, 0.1), psi]),
    )
    t_a = np.sort(rng.uniform(2456900.0, 2457900.0, 12))
    np.savetxt(
        tmp_dir / "a.astrom",
        np.column_stack(
            [
                t_a,
                np.full(12, 120.0),
                np.full(12, -30.0),
                *[np.full(12, 0.2)] * 2,
            ]
        ),
    )
    t_r = np.sort(rng.uniform(2456900.0, 2457900.0, 15))
    # PAs straddling the +/-180 deg wrap
    pa = np.where(np.arange(15) % 2 == 0, 179.0, -179.0)
    np.savetxt(
        tmp_dir / "r.astrom",
        np.column_stack(
            [t_r, np.full(15, 2.0), np.full(15, 0.05), pa, np.full(15, 0.5)]
        ),
    )


def _config(tmp_dir, smear=True):
    extra = {"exptime_s": EXPTIME_S, "ninterp": NINTERP} if smear else {}
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "B"}],
        "orbit": [{"name": "B"}],
        "astrometryinstrument": [
            {
                "name": "G",
                "file": str(tmp_dir / "g.astrom"),
                "mode": "gaia",
                "observer_location": "earth",
                "epoch": EPOCH,
                **extra,
            },
            {
                "name": "Abs",
                "file": str(tmp_dir / "a.astrom"),
                "mode": "abs",
                "observer_location": "earth",
                "epoch": EPOCH,
                **extra,
            },
            {
                "name": "Rel",
                "file": str(tmp_dir / "r.astrom"),
                "mode": "rel",
                **extra,
            },
        ],
    }


_PARAMS = {
    "star.A.mass": {"initval": 1.0, "sigma": 0.05},
    "star.A.radius": {"initval": 1.0, "sigma": 0.1},
    "star.A.teff": {"initval": 5800, "sigma": 100},
    "star.A.feh": {"initval": 0.0, "sigma": 0.1},
    "star.A.ra": {"initval": 120.0},
    "star.A.dec": {"initval": -30.0},
    "star.A.pm_ra": {"initval": 50.0},
    "star.A.pm_dec": {"initval": -20.0},
    "star.A.distance": {"initval": 10.0},
    "planet.B.mass": {"initval": 500.0},
    "planet.B.radius": {"initval": 1.0, "sigma": 0},
    "orbit.B.period": {"initval": P_D},
    "orbit.B.tc": {"initval": TC},
    "orbit.B.secosw": {"initval": 0.0, "sigma": 0},
    "orbit.B.sesinw": {"initval": 0.0, "sigma": 0},
    "orbit.B.bigomega": {"initval": 40.0},
    "orbit.B.cosi": {"initval": 0.3},
}


def _build(tmp_dir, smear=True):
    system = System(_config(tmp_dir, smear), user_params=_PARAMS)
    system.prepare()
    model = system.build_model()
    return system, model


@pytest.fixture(scope="module")
def smeared(tmp_path_factory):
    d = tmp_path_factory.mktemp("astrom_exposure")
    _write(d)
    system, model = _build(d)
    return d, system, model


def _vals(system):
    """The plot functions' parameter values: the start (initvals)."""
    return system.astrometryinstrument._point_values(system, {})


def _at_vals(system, nodes, vals):
    """Evaluate likelihood nodes with ``system.plot_params`` set to
    ``vals`` -- the inputs the compiled plot functions take -- so the plot
    path and the likelihood graph see one and the same point."""
    import pytensor

    fn = pytensor.function(
        [p.value for p in system.plot_params],
        list(nodes),
        on_unused_input="ignore",
    )
    return [np.asarray(x) for x in fn(*vals)]


def _simpson_nodes(t, n=401):
    """Composite Simpson nodes/weights over each exposure (rows of t)."""
    u = np.linspace(-0.5, 0.5, n)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w /= w.sum()
    return t[:, None] + u[None, :] * EXPTIME_S / 86400.0, w


def _instant_abs(ai, system, i, tt, vals, point):
    """Instantaneous (dE, dN) of absolute dataset i at times tt (any shape),
    from the plot path's own instantaneous pieces."""
    from exozippy.ephemeris import observer_position_at_reception
    from exozippy.skyframe import parallax_factors

    d = ai.datasets[i]
    flat = tt.ravel()
    xyz = observer_position_at_reception(
        flat, d["ra_ref"], d["dec_ref"], observer_location=ai.observers[i]
    )
    P_E, P_N = parallax_factors(xyz, d["ra_ref"], d["dec_ref"])
    lE, lN = ai._linear_terms(dict(d, P_E=P_E, P_N=P_N), flat, point, system)
    oE, oN = ai._eval_photo(i, flat, vals)
    return (lE + oE).reshape(tt.shape), (lN + oN).reshape(tt.shape)


def test_rel_smeared_position_is_the_averaged_track(smeared):
    """
    Given a rel dataset of a 1-day orbit with 6-hour exposures,
    When the likelihood's exposure-averaged (dE, dN) is evaluated,
    Then it equals a dense numerical average of the instantaneous relative
    track over each exposure, and the separation formed from it differs
    measurably from the averaged separation -- the reason the vector, not
    the observable, is what gets averaged.
    """
    _, system, model = smeared
    ai = system.astrometryinstrument
    i = 2
    t = ai.datasets[i]["time"]
    vals = _vals(system)
    dE, dN = _at_vals(system, ai._data_position_nodes[i], vals)

    tt, w = _simpson_nodes(t)
    rE, rN = ai._compiled_rel[i](tt.ravel(), *vals)
    rE, rN = rE.reshape(tt.shape), rN.reshape(tt.shape)
    amp = np.max(np.hypot(rE, rN))
    np.testing.assert_allclose(dE, rE @ w, rtol=0, atol=1e-9 * amp)
    np.testing.assert_allclose(dN, rN @ w, rtol=0, atol=1e-9 * amp)

    # the smearing is real here, and averaging sep would be wrong
    mid_E, mid_N = ai._compiled_rel[i](t, *vals)
    assert np.max(np.hypot(dE - mid_E, dN - mid_N)) > 1e-2 * amp
    assert np.max(np.abs(np.hypot(rE, rN) @ w - np.hypot(dE, dN))) > 1e-3 * amp


def test_gaia_along_scan_of_the_smeared_vector_is_the_smeared_abscissa(
    smeared,
):
    """
    Given a gaia dataset with 6-hour exposures,
    When the along-scan model is formed from the likelihood's averaged
    vector,
    Then it equals the node-weighted average of the instantaneous along-scan
    abscissa at the same Gauss nodes (the projection is linear), and the
    averaged vector matches a dense numerical average of the track.
    """
    _, system, model = smeared
    ai = system.astrometryinstrument
    i = 0
    d = ai.datasets[i]
    vals = _vals(system)
    dE, dN = _at_vals(system, ai._data_position_nodes[i], vals)
    w_smeared = dE * d["sin_psi"] + dN * d["cos_psi"]

    offsets, wq = ai.exposure_nodes(i)
    tq = d["time"][:, None] + offsets[None, :]
    qE, qN = _instant_abs(ai, system, i, tq, vals, {})
    w_nodes = qE * d["sin_psi"][:, None] + qN * d["cos_psi"][:, None]
    amp = np.max(np.abs(w_nodes))
    np.testing.assert_allclose(
        w_smeared, w_nodes @ wq, rtol=0, atol=1e-9 * amp
    )

    tt, ws = _simpson_nodes(d["time"])
    sE, sN = _instant_abs(ai, system, i, tt, vals, {})
    orb = 2.0  # mas scale of the photocenter wobble at this geometry
    np.testing.assert_allclose(dE, sE @ ws, rtol=0, atol=1e-6 * orb)
    np.testing.assert_allclose(dN, sN @ ws, rtol=0, atol=1e-6 * orb)


def test_the_plotted_model_is_the_likelihood_model(smeared):
    """
    Given smeared gaia, abs and rel datasets,
    When the plot path evaluates each dataset's model at its own epochs
    (the compiled exposure model plot_data and plot_sky call),
    Then it equals the likelihood's averaged (dE, dN), and the gaia chart's
    model trace equals the likelihood's along-scan model.
    """
    _, system, model = smeared
    ai = system.astrometryinstrument
    vals = _vals(system)
    for i, d in enumerate(ai.datasets):
        dE, dN = _at_vals(system, ai._data_position_nodes[i], vals)
        pE, pN = ai._eval_exposure(system, i, d["time"], vals)
        np.testing.assert_allclose(pE, dE, rtol=0, atol=1e-9)
        np.testing.assert_allclose(pN, dN, rtol=0, atol=1e-9)

    d = ai.datasets[0]
    dE, dN = _at_vals(system, ai._data_position_nodes[0], vals)
    chart = next(c for c in ai.plot_data(system, {}) if c.id.endswith(".G"))
    trace = next(tr for tr in chart.traces if tr.name == "model")
    np.testing.assert_allclose(
        trace.y, dE * d["sin_psi"] + dN * d["cos_psi"], rtol=0, atol=1e-9
    )


def test_smeared_logp_is_finite_across_the_pa_wrap(smeared):
    """
    Given rel data whose PAs straddle +/-180 deg and smeared exposures,
    When the start logp and gradient are evaluated,
    Then both are finite: the PA comes from the averaged vector, so no
    average is ever taken across the wrap.
    """
    _, _, model = smeared
    ip = model.initial_point(random_seed=0)
    assert np.isfinite(model.compile_logp()(ip))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))


def test_ninterp_one_is_the_instantaneous_model(smeared, tmp_path):
    """
    Given the same datasets with exptime_s stated but ninterp 1, and with
    no exposure keys at all,
    When the start logps are compared,
    Then they are identical: one node is the timestamp itself.
    """
    d, _, _ = smeared
    cfg = _config(d, smear=False)
    for entry in cfg["astrometryinstrument"]:
        entry["exptime_s"] = EXPTIME_S
        entry["ninterp"] = 1
    a = System(cfg, user_params=_PARAMS)
    a.prepare()
    ma = a.build_model()
    _, mb = _build(d, smear=False)
    assert ma.compile_logp()(
        ma.initial_point(random_seed=0)
    ) == mb.compile_logp()(mb.initial_point(random_seed=0))


def test_a_catalog_dataset_refuses_the_exposure_keys(tmp_path):
    """
    Given a catalog-mode dataset with ninterp,
    When the system is constructed,
    Then it raises naming the dataset: a catalog solution is not a time
    series of exposures.
    """
    cfg = {
        "star": [{"name": "A", "mist": False}],
        "astrometryinstrument": [
            {
                "name": "Hip",
                "file": "unused.cat",
                "mode": "catalog",
                "ninterp": 3,
                "exptime_s": 600.0,
            }
        ],
    }
    with pytest.raises(
        ValueError, match=r"astrometryinstrument\.Hip.*catalog"
    ):
        System(cfg, user_params={})
