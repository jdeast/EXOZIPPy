"""tc is SAMPLED near the data and REPORTED at two epochs (review 2.14.9).

JDE 2026-10-07: "sample tc near the data's center and report it at the
user's epoch.  We also want to report it at the optimal epoch, as EXOFASTv2
does."  A conjunction quoted far from the data -- `examples/kelt4`'s TESS-era
seed is ~1160 periods after its RVs -- is correlated with the period almost
perfectly (the item measured corr 0.94 and raw stds ~20 on kelt4_rvonly),
which costs the sampler several times the gradient evaluations.  So:

  * the SAMPLED conjunction is the one nearest the data's time center
    (`Orbit._sampling_epochs`, at stage 3, from `epochs_constraining`), and
    an orbit already near its data keeps exactly the graph it always had;
  * `tc` stays the conjunction at the USER's epoch, derived from the sampled
    one, so the period's uncertainty propagates into it;
  * `t0` is reported at the epoch the posterior itself prefers (EXOFASTv2's
    T_0, `Parameter.shift_to_optimal_epoch`);
  * the restart file writes `tc` at the user's epoch, so the next fit means
    the same thing (`restart_as`);
  * the node-degeneracy fold moves the SAMPLED conjunction.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import yaml

from exozippy import trace_meta
from exozippy.components.orbit import Orbit, physics
from exozippy.config import ConfigManager
from exozippy.system import System

P = 3.0
TC_TRUE = 2455200.0
# The user's seed: the same conjunction 500 periods after the data.
N_FAR = 500
TC_FAR = TC_TRUE + N_FAR * P


class _Epochs:
    """A stand-in data component: fixed epochs on fixed orbits."""

    def __init__(self, per_orbit, rank=1):
        self.per_orbit = per_orbit
        self.epoch_timing_rank = rank

    def epochs_constraining(self, system, orbit):
        return {
            o: [(np.asarray(t, float), np.ones(len(t)))]
            for o, t in self.per_orbit.items()
        }


def _registered(user_params, names=("b",), data=None, eclipses=None):
    """An Orbit after stage 3, with `data` (smooth-curve epochs, rank 1)
    and `eclipses` (rank 2) as its data components."""
    cm = ConfigManager(dict(user_params))
    orbit = Orbit([{"name": n} for n in names], cm)
    comps = {"orbit": orbit}
    if data is not None:
        comps["data"] = _Epochs(data)
    if eclipses is not None:
        comps["eclipses"] = _Epochs(eclipses, rank=2)
    orbit.register_parameters(SimpleNamespace(active_components=comps))
    return orbit, cm


# ---------------------------------------------------------------------------
# 1. The epoch choice (stage 3)
# ---------------------------------------------------------------------------


def test_a_seed_far_from_the_data_is_sampled_near_them():
    """
    Given a tc seed 500 periods after every data epoch,
    When the orbit registers its parameters,
    Then it samples `tc_sampled`, the conjunction nearest the data's time
      center, in a one-period window around it, and derives `tc` (at the
      user's epoch) from it with no bound of its own.
    """
    t = np.linspace(TC_TRUE - 40.0, TC_TRUE + 40.0, 30)
    orbit, cm = _registered(
        {"orbit.b.period": {"initval": P}, "orbit.b.tc": {"initval": TC_FAR}},
        data={0: t},
    )

    assert orbit.tc_epoch.tolist() == [-N_FAR]
    np.testing.assert_allclose(orbit.data_time_center, [t.mean()])
    entry = orbit.manifest["tc_sampled"]
    np.testing.assert_allclose(entry["lower"], [TC_TRUE - P / 2])
    np.testing.assert_allclose(entry["upper"], [TC_TRUE + P / 2])
    tc = orbit.manifest["tc"]
    assert tc["expr_key"] == "from_sampled" and tc["force_node"]
    assert np.isneginf(tc["lower"]).all() and np.isposinf(tc["upper"]).all()
    # The start of the moved conjunction is the user's orbit, moved.
    assert cm.hints["orbit.0.tc_sampled"] == pytest.approx(TC_TRUE)


def test_a_seed_near_the_data_keeps_the_manifest_it_always_had():
    """
    Given a tc seed within half a period of the data's time center,
    When the orbit registers its parameters,
    Then nothing moves: `tc` is sampled in the window it always had and
      there is no `tc_sampled` -- which is why every shipped example seeded
      near its data starts bit-identically.
    """
    t = np.linspace(TC_TRUE - 40.0, TC_TRUE + 40.0, 30)
    orbit, _ = _registered(
        {
            "orbit.b.period": {"initval": P},
            "orbit.b.tc": {"initval": TC_TRUE + 1.0},
        },
        data={0: t},
    )

    assert orbit.tc_epoch.tolist() == [0]
    assert "tc_sampled" not in orbit.manifest
    assert orbit.manifest["tc"] == {
        "force_node": True,
        "lower": pytest.approx([TC_TRUE + 1.0 - P / 2]),
        "upper": pytest.approx([TC_TRUE + 1.0 + P / 2]),
    }


def test_a_seed_inside_the_data_span_is_the_users_choice():
    """
    Given a tc seed near one end of its data, many periods from their
      center but inside their span,
    When the orbit registers its parameters,
    Then it is not moved: the optimal epoch lies inside the span too, and
      the equal-weight center is no better a guess than the user's own --
      measured worse on examples/kelt17 (corr(tc, P) -0.70 at the center,
      -0.41 at the seed) and examples/gj1214 (-0.20 against -0.06).
    """
    t = np.linspace(TC_TRUE - 40.0, TC_TRUE + 40.0, 30)
    orbit, _ = _registered(
        {
            "orbit.b.period": {"initval": P},
            "orbit.b.tc": {"initval": TC_TRUE + 39.0},
        },
        data={0: t},
    )
    assert orbit.tc_epoch.tolist() == [0]
    assert "tc_sampled" not in orbit.manifest


def test_eclipses_set_the_center_over_rvs():
    """
    Given one orbit timed by three transits around the seed and by a long
      baseline of RVs months earlier,
    When the orbit registers its parameters,
    Then only the eclipses set the center.  On `examples/kelt17`'s fast
      config (two eclipses, twelve RVs) an equal-weight mean over both
      dragged the sampled epoch two periods from the seed, where the
      measured tc-P correlation was -0.98 against +0.30 at the seed.
    """
    rv = np.linspace(TC_TRUE - 300.0, TC_TRUE - 100.0, 400)
    transits = np.concatenate(
        [
            np.linspace(TC_TRUE + k * P - 0.1, TC_TRUE + k * P + 0.1, 30)
            for k in (-1, 0, 1)
        ]
    )
    orbit, _ = _registered(
        {"orbit.b.period": {"initval": P}, "orbit.b.tc": {"initval": TC_TRUE}},
        data={0: rv},
        eclipses={0: transits},
    )
    assert orbit.tc_epoch.tolist() == [0]
    np.testing.assert_allclose(orbit.data_time_center, [transits.mean()])


def test_a_periastron_seed_is_moved_like_a_conjunction_seed():
    """
    Given a time of PERIASTRON seeded far from the data, and no tc,
    When the orbit registers its parameters,
    Then the user's epoch is the conjunction that periastron implies
      (`_seeded_tc`, review 8.1.1), and the sampled one is that conjunction
      moved to the data -- the tp channel and the move compose.
    """
    t = np.linspace(TC_TRUE - 40.0, TC_TRUE + 40.0, 30)
    sc, ss = 0.3, 0.4
    tp = TC_FAR - 0.37
    orbit, cm = _registered(
        {
            "orbit.b.period": {"initval": P},
            "orbit.b.tp": {"initval": tp},
            "orbit.b.secosw": {"initval": sc},
            "orbit.b.sesinw": {"initval": ss},
        },
        data={0: t},
    )
    implied = float(
        physics.tc_from_tp(
            np.array([tp]),
            np.array([sc**2 + ss**2]),
            np.array([np.arctan2(ss, sc)]),
            np.array([P]),
        )[0]
    )
    n = int(np.round((t.mean() - implied) / P))
    assert n != 0
    assert orbit.tc_epoch.tolist() == [n]
    assert cm.hints["orbit.0.tc_sampled"] == pytest.approx(implied + n * P)


def test_an_orbit_no_data_times_is_not_moved():
    """
    Given an orbit no dataset constrains,
    When it registers its parameters,
    Then its sampled epoch is the user's: there is nothing to be near.
    """
    orbit, _ = _registered(
        {"orbit.b.period": {"initval": P}, "orbit.b.tc": {"initval": TC_FAR}}
    )
    assert orbit.tc_epoch.tolist() == [0]
    assert np.isnan(orbit.data_time_center).all()
    assert "tc_sampled" not in orbit.manifest


@pytest.mark.parametrize(
    "extra",
    [
        {"sigma": 0},
        {"lower": TC_FAR - 1.0},
        {"upper": TC_FAR + 1.0},
    ],
    ids=["pinned", "lower", "upper"],
)
def test_a_user_pin_or_bound_on_tc_keeps_the_user_epoch(extra):
    """
    Given a user who pinned or bounded `tc` itself,
    When the orbit registers its parameters,
    Then it is sampled at the user's epoch: a pin on a derived element is
      dropped and a bound on one becomes a soft barrier, so moving the
      sampled epoch would silently change what the user wrote.
    """
    t = np.linspace(TC_TRUE - 40.0, TC_TRUE + 40.0, 30)
    # The index spelling: a System standardizes every element name to it
    # before stage 3 (config.md), and this harness has no System.
    orbit, _ = _registered(
        {
            "orbit.b.period": {"initval": P},
            "orbit.0.tc": {"initval": TC_FAR, **extra},
        },
        data={0: t},
    )
    assert orbit.tc_epoch.tolist() == [0]
    assert "tc_sampled" not in orbit.manifest


def test_the_choice_is_per_orbit():
    """
    Given two orbits, one seeded far from its data and one near,
    When they register their parameters,
    Then only the far one moves: `tc` is derived on that element alone and
      `tc_sampled` is a parameter of that element alone.
    """
    t = np.linspace(TC_TRUE - 40.0, TC_TRUE + 40.0, 30)
    orbit, _ = _registered(
        {
            "orbit.near.period": {"initval": P},
            "orbit.near.tc": {"initval": TC_TRUE},
            "orbit.far.period": {"initval": P},
            "orbit.far.tc": {"initval": TC_FAR},
        },
        names=("near", "far"),
        data={0: t, 1: t},
    )
    assert orbit.tc_epoch.tolist() == [0, -N_FAR]
    tc = orbit.manifest["tc"]
    np.testing.assert_array_equal(
        tc["expr_key"]["from_sampled"], [False, True]
    )
    np.testing.assert_array_equal(
        orbit.manifest["tc_sampled"]["mask"], [False, True]
    )


# ---------------------------------------------------------------------------
# 2. A real system: the start, the fold
# ---------------------------------------------------------------------------


def _write_rv(path, tc=TC_TRUE, seed=3):
    rng = np.random.default_rng(seed)
    # Symmetric about tc, so the data's time center IS tc.
    t = np.linspace(tc - 60.3, tc + 60.3, 40)
    rv = -60.0 * np.sin(2.0 * np.pi * (t - tc) / P) + rng.normal(
        0, 5.0, t.size
    )
    np.savetxt(path, np.column_stack([t, rv, np.full(t.size, 5.0)]))
    return str(path)


def _rv_params(tc):
    return {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "planet.b.mass": {"initval": 0.5},
        "orbit.b.period": {"initval": P},
        "orbit.b.tc": {"initval": tc},
        "orbit.b.secosw": {"initval": 0.1},
        "orbit.b.sesinw": {"initval": 0.1},
        "orbit.b.cosi": {"initval": 0.1},
    }


def _rv_config(d):
    """The config, with the params file it names written beside it (mkparam
    re-hashes the two, so they must be what the System was built from)."""
    (d / "rv.params.yaml").write_text(yaml.safe_dump(_rv_params(TC_FAR)))
    return {
        "prefix": str(d / "fit"),
        "parameter_file": str(d / "rv.params.yaml"),
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "rvinstrument": [{"name": "inst", "file": _write_rv(d / "rv.dat")}],
    }


@pytest.fixture(scope="module")
def far_seed(tmp_path_factory):
    d = tmp_path_factory.mktemp("tc_epoch_rv")
    config = _rv_config(d)
    system = System(config, _rv_params(TC_FAR))
    system.prepare()
    model = system.build_model()
    return system, model, config, d


def _start_values(system, model, names):
    """What the built graph computes for orbit parameters at the start
    (internal units; days for every name used here).  RVs replaced by their
    value variables -- `.eval()` would draw from the prior."""
    import pytensor

    start = system.get_raw_start(model)
    nodes = [getattr(system.orbit, n).value for n in names]
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values(nodes),
        on_unused_input="ignore",
    )
    out = fn(*[start[v.name] for v in model.value_vars])
    return {n: np.atleast_1d(np.asarray(o, float)) for n, o in zip(names, out)}


def test_the_build_start_is_the_orbit_the_user_seeded(far_seed):
    """
    Given a tc seed 500 periods from the data,
    When the model is built,
    Then `tc` -- derived, at the user's epoch -- starts at the seed, and the
      sampled `tc_sampled` at the same conjunction 500 periods earlier: the
      same physical orbit, so the start logp is the one the user's seed
      always gave (measured on the shipped kelt4_rvonly: -601.0870105 vs
      -601.0870104 before, the rounding of `tc_sampled - N P`).
    """
    system, model, _, _ = far_seed
    assert system.orbit.tc_epoch.tolist() == [-N_FAR]
    v = _start_values(system, model, ("tc", "tc_sampled", "t0"))
    np.testing.assert_allclose(v["tc"], [TC_FAR], rtol=0, atol=1e-7)
    np.testing.assert_allclose(v["tc_sampled"], [TC_TRUE], rtol=0, atol=1e-7)
    # t0's NODE is the sampled epoch; its report moves after sampling.
    np.testing.assert_allclose(v["t0"], [TC_TRUE], rtol=0, atol=1e-7)


def test_the_shifted_model_starts_where_a_near_seed_does(far_seed, tmp_path):
    """
    Given the same data seeded at the far epoch and at the near one,
    When both models are built,
    Then their start logps agree: the move is a relabelling of the same
      conjunction, and the one-period window around it has the same width.
    """
    system, model, config, _ = far_seed
    near = System(config, _rv_params(TC_TRUE))
    near.prepare()
    near_model = near.build_model()
    assert near.orbit.tc_epoch.tolist() == [0]

    lp_far = float(model.compile_logp()(system.get_raw_start(model)))
    lp_near = float(near_model.compile_logp()(near.get_raw_start(near_model)))
    assert lp_far == pytest.approx(lp_near, abs=1e-6)


def test_the_fold_moves_the_sampled_conjunction(tmp_path):
    """
    Given an astrometry-only (node-degenerate) orbit seeded 10 periods from
      its data, and a posterior with one draw in each node label,
    When the fold runs,
    Then it rewrites `tc_sampled` -- the coordinate the sampler moved --
      and both draws land on the same label inside its window.
    """
    # tests/test_node_degeneracy.py's degenerate system, seeded 10 periods
    # before its data.
    period, tc_true = 400.0, 2455100.0
    t = np.linspace(2455000.0, 2455000.0 + 2.0 * period, 24)
    phase = 2 * np.pi * (t - tc_true) / period
    noise = np.random.default_rng(11)
    sep = 20.0 + 5.0 * np.cos(phase) + noise.normal(0, 0.2, t.size)
    pa = np.degrees(np.arctan2(np.sin(phase), 0.6 * np.cos(phase))) % 360.0
    pa = pa + noise.normal(0, 0.5, t.size)
    rel = tmp_path / "sim.rel.astrom"
    np.savetxt(
        rel,
        np.column_stack(
            [t, sep, np.full(t.size, 0.2), pa, np.full(t.size, 0.5)]
        ),
    )
    ecc, om = 0.35, np.radians(70.0)
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.distance": {"initval": 100.0},
        "planet.BH.mass": {"initval": 0.3 * 1047.5655},
        "planet.BH.radius": {"initval": 1.0, "sigma": 0},
        "orbit.BH.period": {"initval": period},
        "orbit.BH.tc": {"initval": tc_true - 10 * period},
        "orbit.BH.secosw": {"initval": np.sqrt(ecc) * np.cos(om)},
        "orbit.BH.sesinw": {"initval": np.sqrt(ecc) * np.sin(om)},
        "orbit.BH.bigomega": {"initval": 150.0},
        "orbit.BH.cosi": {"initval": np.cos(np.radians(65.0))},
    }
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [{"name": "BH"}],
        "astrometryinstrument": [
            {"name": "Rel", "file": str(rel), "mode": "rel"}
        ],
    }
    system = System(config, params)
    system.prepare()
    model = system.build_model()
    orbit = system.orbit
    assert orbit.tc_epoch[0] != 0 and orbit.node_degenerate[0]

    point = model.initial_point()
    names = (
        "xbigomega",
        "ybigomega",
        "secosw",
        "sesinw",
        "tc_sampled",
        "logP",
    )

    def raw(name, pt_):
        return pt_[f"orbit.{name}_raw"][0]

    phys = {
        n: getattr(orbit, n).element_phys_from_raw(0, raw(n, point))
        for n in names
    }
    ecc = phys["secosw"] ** 2 + phys["sesinw"] ** 2
    omega = np.arctan2(phys["sesinw"], phys["secosw"])
    delta = (
        physics.mean_anomaly_at_conjunction(ecc, omega + np.pi)
        - physics.mean_anomaly_at_conjunction(ecc, omega)
    ) * (10.0 ** phys["logP"] / (2.0 * np.pi))
    tf = orbit.tc_sampled._raw_transform
    lo, up = float(tf["lowers"][0]), float(tf["uppers"][0])
    partner = dict(phys)
    for n in names[:4]:
        partner[n] = -phys[n]
    partner["tc_sampled"] = lo + np.mod(
        phys["tc_sampled"] + delta - lo, up - lo
    )

    posterior = xr.Dataset(
        {
            f"orbit.{n}_raw": (
                ("chain", "draw", f"orbit.{n}_raw_dim_0"),
                np.array(
                    [
                        [
                            [raw(n, point)],
                            [
                                getattr(orbit, n).element_raw_from_phys(
                                    0, partner[n]
                                )
                            ],
                        ]
                    ]
                ),
            )
            for n in names
        }
    )

    assert orbit.fold_node_degeneracy(posterior)
    for n in names:
        got = posterior[f"orbit.{n}_raw"].values[0, :, 0]
        np.testing.assert_allclose(got[0], got[1], atol=1e-7)
    tc_s = orbit.tc_sampled.element_phys_from_raw(
        0, posterior["orbit.tc_sampled_raw"].values[0, :, 0]
    )
    assert np.all((tc_s > lo) & (tc_s < up))


# ---------------------------------------------------------------------------
# 3. The optimal epoch (post-sampling)
# ---------------------------------------------------------------------------


def _exofast_scan(tc, period, lo=-2000, hi=2000):
    """EXOFASTv2 derivepars.pro's brute-force scan over integer epochs."""
    best, best_corr = None, np.inf
    for e in range(lo, hi + 1):
        c = abs(np.corrcoef(tc + e * period, period)[0, 1])
        if c < best_corr:
            best, best_corr = e, c
    return best


@pytest.mark.parametrize("true_epoch", [0, 7, -1164])
def test_the_optimal_epoch_is_exofastv2s(far_seed, true_epoch):
    """
    Given draws of a conjunction quoted `true_epoch` periods from the epoch
      where it decorrelates from the period,
    When the posterior is moved to the optimal epoch,
    Then the shift is the integer EXOFASTv2's derivepars.pro scan picks,
      every draw moves by that many of its OWN periods, and the result is
      uncorrelated with the period.
    """
    system, _, _, _ = far_seed
    orbit = system.orbit
    rng = np.random.default_rng(1)
    n = 4000
    per = P + 2e-6 * rng.standard_normal(n)
    t_opt = TC_TRUE + 1e-4 * rng.standard_normal(n)
    drawn = t_opt - true_epoch * per  # the same conjunction, far away
    orbit.period.posterior = per[None, :]
    orbit.t0.posterior = drawn[None, :]

    orbit.t0.shift_to_optimal_epoch(system.get_parameter_lookup())

    assert orbit.t0.optimal_epoch_shift.tolist() == [_exofast_scan(drawn, per)]
    assert orbit.t0.optimal_epoch_shift.tolist() == [true_epoch]
    got = np.asarray(orbit.t0.posterior)[0]
    np.testing.assert_allclose(
        got, drawn + true_epoch * per, rtol=0, atol=1e-6
    )
    assert abs(np.corrcoef(got, per)[0, 1]) < 0.05


def test_a_pinned_period_moves_nothing(far_seed):
    """
    Given a period with no spread,
    When the posterior is moved to the optimal epoch,
    Then nothing moves: every epoch is equally good.
    """
    system, _, _, _ = far_seed
    orbit = system.orbit
    orbit.period.posterior = np.full((1, 100), P)
    draws = TC_TRUE + 1e-4 * np.random.default_rng(2).standard_normal(100)
    orbit.t0.posterior = draws[None, :]
    orbit.t0.shift_to_optimal_epoch(system.get_parameter_lookup())
    assert orbit.t0.optimal_epoch_shift.tolist() == [0]
    np.testing.assert_array_equal(np.asarray(orbit.t0.posterior)[0], draws)


# ---------------------------------------------------------------------------
# 4. The restart file
# ---------------------------------------------------------------------------


def _one_draw_trace(system, model, path):
    """A one-draw trace at the build start, stamped like a real one."""
    import arviz as az
    import pymc as pm

    start = system.get_raw_start(model)
    raws = {
        k: np.atleast_1d(np.asarray(v, float))
        for k, v in start.items()
        if k.endswith("_raw")
    }
    posterior = xr.Dataset(
        {
            k: (("chain", "draw", f"{k}_dim_0"), v[None, None, :])
            for k, v in raws.items()
        }
    )
    det = pm.compute_deterministics(
        posterior, model=model, extend_dataset=True, progressbar=False
    )
    idata = az.from_dict(
        {
            "posterior": det,
            "sample_stats": xr.Dataset(
                {
                    "lp": xr.DataArray(
                        np.array([[-10.0]]), dims=["chain", "draw"]
                    )
                }
            ),
        }
    )
    trace_meta.apply_metadata(idata, trace_meta.structural_metadata(system))
    idata.to_netcdf(str(path))
    return path


def test_the_restart_file_writes_tc_at_the_users_epoch(far_seed):
    """
    Given a finished fit that sampled `tc_sampled` near the data,
    When mkparam writes the restart file,
    Then it carries `tc` at the USER's epoch and no `tc_sampled`, and the
      next fit built from it chooses the same sampled epoch and starts at
      the same sampled conjunction -- the file means what it meant.
    """
    from exozippy.mkparam import write_param_file

    system, model, config, d = far_seed
    assert json.loads(
        trace_meta.structural_metadata(system)[trace_meta.RESTART_AS_ATTR]
    ) == {"orbit.tc_sampled": "orbit.tc"}

    trace = _one_draw_trace(system, model, d / "fit_trace.nc")

    out = write_param_file(
        dict(config),
        base_dir=str(d),
        trace_path=str(trace),
        output_path=str(d / "restart.params.yaml"),
    )
    written = yaml.safe_load(open(out))

    assert "orbit.b.tc_sampled" not in written
    assert written["orbit.b.tc"]["initval"] == pytest.approx(TC_FAR, abs=1e-6)

    again = System(dict(config), written)
    again.prepare()
    again_model = again.build_model()
    assert again.orbit.tc_epoch.tolist() == [-N_FAR]
    v = _start_values(again, again_model, ("tc", "tc_sampled"))
    # tc reproduces exactly what was written; tc_sampled to the 8-decimal
    # rounding mkparam applies to logP, times 500 periods.
    np.testing.assert_allclose(
        v["tc"], [written["orbit.b.tc"]["initval"]], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(v["tc_sampled"], [TC_TRUE], rtol=0, atol=1e-4)


def test_a_trace_without_the_sampled_coordinate_is_refused(far_seed):
    """
    Given a trace that sampled `orbit.tc` (older code) for a model that now
      samples `orbit.tc_sampled`,
    When it is reused,
    Then the reuse raises, naming the missing variable and the remedy,
      rather than decoding the model's raws from nothing.
    """
    _, model, _, _ = far_seed
    idata = SimpleNamespace(
        posterior=xr.Dataset({"orbit.tc_raw": ("x", np.zeros(1))})
    )
    with pytest.raises(trace_meta.StaleTraceError, match="tc_sampled_raw"):
        trace_meta.check_trace_coordinates(idata, model, "old_trace.nc")
