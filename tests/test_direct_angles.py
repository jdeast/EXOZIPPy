"""A precisely measured angle is sampled DIRECTLY, automatically (exozippy.angles).

JDE 2026-11-04: "Do the angle reparameterization automatically with a warning
and user override."  An angle behind direction latents (orbit
xbigomega/ybigomega, xomega/yomega; lens xalpha/yalpha) that the data pin is
a narrow ray from the origin in latent space -- the Gaia BH3 node, 0.13 deg,
sampled at max tree depth with chains trapped in the r -> 0 funnel.  Below
`DIRECT_ANGLE_MAX_SIGMA` the angle itself is sampled, uniform on a window
centered on the start.

The synthetic systems here make the angle's information EXACT: relative
astrometry with errors so large the likelihood is flat, plus a user Gaussian
prior on the node, so the probe has a known answer (the prior's sigma) and
"well measured" vs "poorly measured" is one number in the params file.
"""

import logging

import numpy as np
import pytest

from exozippy import angles
from exozippy.system import System
from exozippy.trace_meta import ANGLES_ATTR, structural_metadata

pytestmark = pytest.mark.slow

_DEG = np.pi / 180.0
_T0 = 2455000.0


def _write_rel(path):
    """Relative astrometry whose errors make its likelihood ~flat."""
    t = np.linspace(_T0, _T0 + 800.0, 12)
    phase = 2 * np.pi * (t - _T0) / 400.0
    sep = 20.0 + 5.0 * np.cos(phase)
    pa = np.degrees(np.arctan2(np.sin(phase), 0.6 * np.cos(phase))) % 360.0
    np.savetxt(
        path,
        np.column_stack(
            [t, sep, np.full(t.size, 1e4), pa, np.full(t.size, 1e4)]
        ),
    )
    return str(path)


def _write_rv(path):
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(_T0, _T0 + 400.0, 20))
    np.savetxt(
        path,
        np.column_stack(
            [t, 50.0 * np.sin(2 * np.pi * t / 31.0), np.full(t.size, 1e4)]
        ),
    )
    return str(path)


def _system(tmp, node_sigma_deg, rv=True, orbit_extra=None, extra=None):
    """Build-ready System: the node's information is a Gaussian prior."""
    orbit = {"name": "BH"}
    orbit.update(orbit_extra or {})
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [orbit],
        "astrometryinstrument": [
            {
                "name": "Rel",
                "file": _write_rel(tmp / "rel.astrom"),
                "mode": "rel",
            }
        ],
    }
    if rv:
        config["rvinstrument"] = [
            {"name": "inst", "file": _write_rv(tmp / "rv.dat")}
        ]
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.distance": {"initval": 100.0},
        "planet.BH.mass": {"initval": 300.0},
        "planet.BH.radius": {"initval": 1.0, "sigma": 0},
        "orbit.BH.period": {"initval": 400.0},
        "orbit.BH.tc": {"initval": _T0 + 100.0},
        "orbit.BH.secosw": {"initval": 0.2},
        "orbit.BH.sesinw": {"initval": 0.5},
        "orbit.BH.cosi": {"initval": 0.4},
        "orbit.BH.bigomega": {
            "initval": 150.0,
            "mu": 150.0,
            "sigma": node_sigma_deg,
        },
    }
    params.update(extra or {})
    system = System(config, params)
    system.prepare()
    return system


def _latent_terms(system, raw_start):
    """ln p of the two latents' prior terms at raw_start (see the Jacobian test)."""
    total = 0.0
    for name in ("xbigomega", "ybigomega"):
        par = getattr(system.orbit, name)
        tf = par._raw_transform
        v = float(
            par.element_phys_from_raw(0, raw_start[f"{par.label}_raw"][0])
        )
        q = (v - tf["lowers"][0]) / (tf["uppers"][0] - tf["lowers"][0])
        total += np.log(q * (1 - q)) - 0.5 * np.log(2 * np.pi) - 0.5 * v * v
    return total


@pytest.fixture(scope="module")
def tight(tmp_path_factory):
    """The node measured to 0.1 deg: the switch fires."""
    system = _system(tmp_path_factory.mktemp("tight"), 0.1)
    latent_model = system.build_model()
    raw = system.get_raw_start(latent_model)
    lpf = latent_model.compile_logp()
    out = {
        "system": system,
        "latent_lp": float(lpf(raw)),
        "latent_terms": _latent_terms(system, raw),
        "latent_vars": {v.name for v in latent_model.free_RVs},
    }
    return out, latent_model, raw, lpf


class _Records(logging.Handler):
    """Collects exozippy log records (caplog is function-scoped)."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture(scope="module")
def switched(tight):
    """The tight system after the automatic decision and the rebuild."""
    out, latent_model, raw, lpf = tight
    system = out["system"]
    handler = _Records()
    log = logging.getLogger("exozippy")
    log.addHandler(handler)
    try:
        choices, _ = angles.choose_direct_angles(
            system, latent_model, raw, lpf
        )
    finally:
        log.removeHandler(handler)
    out["warnings"] = [m for m in handler.messages if "DIRECTLY" in m]
    out["choices"] = choices
    model = angles.adopt_direct_angles(system, latent_model, raw, choices)
    system.recenter_whitening_anchor(model)
    out["model"] = model
    out["raw"] = system.get_raw_start(model)
    out["lp"] = float(model.compile_logp()(out["raw"]))
    return out


# ---------------------------------------------------------------------------
# 1. The automatic switch
# ---------------------------------------------------------------------------


def test_the_probe_measures_the_angles_own_sigma(tight):
    """
    Given a node whose only information is a 0.1 deg Gaussian,
    When the coordinate probe measures it along the circle at the start,
    Then it returns 0.1 deg (to the bisection's 1%) -- the number the switch
      is decided on, not a whitening scale of either latent.
    """
    out, _model, raw, lpf = tight
    (slot,) = angles.find_slots(out["system"])

    theta0, sigma = angles.measure_angle_sigma(slot, raw, lpf)

    assert theta0 == pytest.approx(150.0 * _DEG, abs=1e-9)
    assert sigma == pytest.approx(0.1 * _DEG, rel=0.01)


def test_a_tight_angle_switches_with_a_warning(switched):
    """
    Given the tight node,
    When the automatic decision runs,
    Then bigomega element 0 is chosen for the direct coordinate, centered on
      the start, a full turn wide (RVs identify the node), and ONE warning
      names the element, the measured sigma, the threshold and the override.
    """
    rec = switched["choices"]["orbit.bigomega"]
    assert rec["elements"] == [0]
    assert rec["center"][0] == pytest.approx(150.0 * _DEG, abs=1e-9)
    assert rec["halfwidth"][0] == pytest.approx(np.pi)
    assert rec["forced"] == [False]
    (msg,) = switched["warnings"]
    assert "[orbit.BH] bigomega" in msg
    assert "measured to 0.0999 deg" in msg
    assert "5.73 deg" in msg
    assert "fitbigomega: false" in msg


def test_the_direct_model_samples_the_angle_and_drops_the_latents(switched):
    """
    Given the rebuilt model,
    When its free variables and the node's transform are read,
    Then bigomega has a raw coordinate on [150 - 180, 150 + 180] deg and the
      two latents have none (inactive: not part of this orbit's
      parameterization any more).
    """
    names = {v.name for v in switched["model"].free_RVs}
    assert "orbit.bigomega_raw" in names
    assert "orbit.xbigomega_raw" not in names
    assert "orbit.ybigomega_raw" not in names
    assert names - {"orbit.bigomega_raw"} == switched["latent_vars"] - {
        "orbit.xbigomega_raw",
        "orbit.ybigomega_raw",
    }
    tf = switched["system"].orbit.bigomega._raw_transform
    assert tf["lowers"][0] == pytest.approx((150.0 - 180.0) * _DEG, abs=1e-12)
    assert tf["uppers"][0] == pytest.approx((150.0 + 180.0) * _DEG, abs=1e-12)


def test_equal_physical_angles_give_equal_likelihoods(switched):
    """
    Given the latent model at its start and the direct model at the carried-
      over start (the same physical point),
    When their total logps are compared,
    Then they differ by EXACTLY the prior-coordinate term and nothing else:

        Delta = [ln q_a (1 - q_a) - ln sqrt(2 pi)]
              - sum_{v = x, y} [ln q_v (1 - q_v) - ln sqrt(2 pi) - v^2 / 2]

      the direct angle's logit-uniform Jacobian (q_a = 1/2 at the window
      center) against each latent's logit Jacobian on [-100, 100] plus its
      structural N(0, 1).  Both are the same uniform-angle prior in
      different coordinates; the likelihood, every other prior and the
      node's own Gaussian are identical.
    """
    system = switched["system"]
    node = float(
        system.orbit.bigomega.element_phys_from_raw(
            0, switched["raw"]["orbit.bigomega_raw"][0]
        )
    )
    expected = (np.log(0.25) - 0.5 * np.log(2 * np.pi)) - switched[
        "latent_terms"
    ]

    assert node == pytest.approx(150.0 * _DEG, abs=1e-12)
    assert switched["lp"] - switched["latent_lp"] == pytest.approx(
        expected, abs=1e-8
    )


def test_the_choice_travels_with_the_trace(switched, tmp_path):
    """
    Given the switched system,
    When the trace stamp is built and a fresh System restores it from a file
      carrying only that stamp,
    Then the stamp is the exact choice, and the restored System's manifest
      samples bigomega on the same window -- what a recompute_trace: false
      run needs to decode the draws.
    """
    import json

    import h5py

    stamp = structural_metadata(switched["system"])[ANGLES_ATTR]
    trace = tmp_path / "trace.nc"
    with h5py.File(trace, "w") as f:
        f.attrs[ANGLES_ATTR] = stamp
    fresh = _system(tmp_path, 0.1)

    restored = angles.restore_from_trace(fresh, trace)

    assert restored == json.loads(stamp)
    assert (
        restored["orbit.bigomega"]["center"]
        == (switched["choices"]["orbit.bigomega"]["center"])
    )
    window = fresh.orbit.manifest["bigomega"]["direct_window"]
    assert window["center"][0] == restored["orbit.bigomega"]["center"][0]
    assert not fresh.orbit.manifest["xbigomega"]["mask"][0]


def test_c_and_jax_gradients_are_finite_against_the_window_walls(switched):
    """
    Given the direct model with the node pushed to within e^-25 of either
      window wall (logit coordinate +/-25),
    When dlogp is evaluated on the C backend and through JAX,
    Then every component is finite: the window is the ordinary logit
      transform, with no where-trap and no radicand floor to poison it.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pymc.sampling.jax import get_jaxified_logp

    system, model = switched["system"], switched["model"]
    par = system.orbit.bigomega
    tf = par._raw_transform
    dlogp = model.compile_dlogp()
    jlogp = get_jaxified_logp(model)
    for lq in (-25.0, 25.0):
        point = {k: np.array(v, copy=True) for k, v in switched["raw"].items()}
        point["orbit.bigomega_raw"] = np.array(
            [(lq - tf["logit_q_inits"][0]) / tf["init_scale_logits"][0]]
        )
        assert np.all(np.isfinite(dlogp(point)))
        vals = [point[v.name] for v in model.value_vars]
        grads = jax.grad(lambda vs: jlogp(vs))(vals)
        assert all(np.all(np.isfinite(np.asarray(g))) for g in grads)


# ---------------------------------------------------------------------------
# 2. A loose angle keeps its latents, and the overrides work both ways
# ---------------------------------------------------------------------------


def test_a_loose_angle_keeps_its_latents_and_the_model(tmp_path):
    """
    Given a node known only to 60 deg,
    When the automatic decision runs,
    Then nothing switches: the choices are empty, so run.py keeps the model
      it has (bit-identical start) -- and the probe needed TWO logp calls to
      say so.
    """
    system = _system(tmp_path, 60.0)
    model = system.build_model()
    raw = system.get_raw_start(model)
    calls = []
    lpf = model.compile_logp()

    def counting(point):
        calls.append(1)
        return lpf(point)

    choices, _ = angles.choose_direct_angles(system, model, raw, counting)

    assert choices == {}
    assert len(calls) == 2  # the start, then one step past the threshold
    assert "orbit.xbigomega_raw" in {v.name for v in model.free_RVs}


def test_fit_false_keeps_the_latents_of_a_tight_angle(tmp_path):
    """
    Given the tight node and 'fitbigomega: false' on the orbit block,
    When the automatic decision runs,
    Then nothing switches and no probe is run at all.
    """
    system = _system(tmp_path, 0.1, orbit_extra={"fitbigomega": False})
    model = system.build_model()
    raw = system.get_raw_start(model)

    def never(point):
        raise AssertionError("the probe ran despite fitbigomega: false")

    choices, _ = angles.choose_direct_angles(system, model, raw, never)

    assert choices == {}


def test_fit_true_samples_a_loose_angle_directly_from_the_build(tmp_path):
    """
    Given a loose node and 'fitbigomega: true',
    When the System is prepared and built (no decision step at all -- the GUI
      path),
    Then bigomega is already sampled directly, centered on its resolved start
      (150 deg), with the threshold as its preliminary scale, and the record
      says it was forced.
    """
    system = _system(tmp_path, 60.0, orbit_extra={"fitbigomega": True})
    model = system.build_model()

    rec = system.direct_angles["orbit.bigomega"]
    assert rec["forced"] == [True]
    assert rec["center"][0] == pytest.approx(150.0 * _DEG, abs=1e-9)
    names = {v.name for v in model.free_RVs}
    assert "orbit.bigomega_raw" in names
    assert "orbit.xbigomega_raw" not in names
    assert system.orbit.bigomega.init_scale[0] == pytest.approx(
        angles.DIRECT_ANGLE_MAX_SIGMA
    )


def test_fit_true_with_a_user_bound_raises(tmp_path):
    """
    Given 'fitbigomega: true' and a user lower/upper on bigomega,
    When the System is prepared,
    Then it raises: the direct coordinate's support IS its window.
    """
    with pytest.raises(ValueError, match="support"):
        _system(
            tmp_path,
            60.0,
            orbit_extra={"fitbigomega": True},
            extra={
                "orbit.BH.bigomega": {
                    "initval": 150.0,
                    "lower": 0.0,
                    "upper": 180.0,
                }
            },
        )


def test_a_flag_with_nothing_to_switch_raises(tmp_path):
    """
    Given 'fitomega: true' on an orbit sampling sqrt(e)cos/sin (its omega has
      no direction latents),
    When the System is prepared,
    Then it raises naming the flag, rather than ignoring it.
    """
    with pytest.raises(ValueError, match="fitomega"):
        _system(tmp_path, 60.0, orbit_extra={"fitomega": True})


# ---------------------------------------------------------------------------
# 3. Node degeneracy, seeds
# ---------------------------------------------------------------------------


def test_a_node_degenerate_orbit_gets_half_a_turn_and_no_fold(tmp_path):
    """
    Given astrometry with NO radial data (the node is degenerate with its
      antipode) and 'fitbigomega: true',
    When the System is built and its posterior folded,
    Then the window is [start - 90, start + 90] deg -- one label -- and the
      fold leaves the draws alone (there is no second label to collapse).
    """
    import xarray as xr

    system = _system(
        tmp_path, 60.0, rv=False, orbit_extra={"fitbigomega": True}
    )
    system.build_model()
    tf = system.orbit.bigomega._raw_transform
    assert tf["uppers"][0] - tf["lowers"][0] == pytest.approx(np.pi)
    assert tf["lowers"][0] == pytest.approx(60.0 * _DEG, abs=1e-9)

    raw_draws = np.linspace(-3.0, 3.0, 8).reshape(1, 8, 1)
    posterior = xr.Dataset(
        {
            "orbit.bigomega_raw": xr.DataArray(
                raw_draws.copy(), dims=["chain", "draw", "orbit.bigomega_dim"]
            )
        }
    )
    before = posterior["orbit.bigomega_raw"].values.copy()

    moved = system.orbit.fold_node_degeneracy(posterior)

    assert moved is False
    np.testing.assert_array_equal(
        posterior["orbit.bigomega_raw"].values, before
    )


def test_seeds_that_disagree_by_a_half_turn_keep_the_latents(tmp_path, caplog):
    """
    Given a tight node whose two starting seeds are 150 and 330 deg (a
      reflection apart),
    When the automatic decision runs,
    Then it refuses the switch with a warning: a window centered on seed 0
      would put seed 1 on its seam.
    """
    system = _system(
        tmp_path,
        0.1,
        extra={"orbit.BH.bigomega": {"initval": [150.0, 330.0]}},
    )
    model = system.build_model()
    raw = system.get_raw_start(model)
    with caplog.at_level(logging.WARNING, logger="exozippy"):
        choices, _ = angles.choose_direct_angles(
            system, model, raw, _gaussian_on_node(system, model, 0.1)
        )

    assert choices == {}
    assert any("seeds" in r.getMessage() for r in caplog.records)


def test_seed_starts_carry_their_own_angle_into_the_window(tmp_path):
    """
    Given a tight node with seeds at 150 and 152 deg,
    When the model is switched and the multi-seed starts are built,
    Then seed 1's direct coordinate decodes to 152 deg -- the PTDE start
      population keeps each seed where it was.
    """
    system = _system(
        tmp_path,
        0.1,
        extra={"orbit.BH.bigomega": {"initval": [150.0, 152.0]}},
    )
    model = system.build_model()
    raw = system.get_raw_start(model)
    lpf = _gaussian_on_node(system, model, 0.1)
    choices, _ = angles.choose_direct_angles(system, model, raw, lpf)
    new_model = angles.adopt_direct_angles(system, model, raw, choices)
    system.recenter_whitening_anchor(new_model)

    starts, seeds = system.get_raw_starts(new_model)

    par = system.orbit.bigomega
    decoded = [
        float(par.element_phys_from_raw(0, s["orbit.bigomega_raw"][0]))
        for s in starts
    ]
    assert seeds == [0, 1]
    assert decoded == pytest.approx([150.0 * _DEG, 152.0 * _DEG], abs=1e-9)


def _gaussian_on_node(system, model, sigma_deg):
    """A logp with a Gaussian on the node, for systems whose params file
    spends the bigomega entry on a seed list."""
    import pytensor.tensor as pt

    base = model.compile_logp()
    expr = model.replace_rvs_by_values(
        [pt.flatten(system.orbit.bigomega.value)[0]]
    )[0]
    node = model.compile_fn(
        expr,
        inputs=model.value_vars,
        on_unused_input="ignore",
        point_fn=True,
    )
    center = 150.0 * _DEG
    s = sigma_deg * _DEG

    def logp(point):
        d = np.angle(np.exp(1j * (float(node(point)) - center)))
        return float(base(point)) - 0.5 * (d / s) ** 2

    return logp
