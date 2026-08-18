"""Data-driven whitening: probe + shared-variable rescale (whitening.py,
Parameter.set_whitening).

The model is built with preliminary scales, the probe measures each raw
element's true 0.5-nat step, and set_whitening applies it in place via
pytensor.shared -- the posterior must be untouched, the start must not move,
and a re-probe of the rescaled model must land at ~1.
"""

import numpy as np
import pymc as pm

from exozippy.components.parameter import Parameter
from exozippy.whitening import (
    _CLIP_LO,
    apply_measured_whitening,
    probe_scales,
)


def _toy_model():
    """Two bounded params with very different posterior widths, plus an
    unbounded Gaussian-prior param (whose raw N(0,1) IS the prior)."""
    p_tight = Parameter(label="toy.x", initval=2.0, lower=0.0, upper=10.0)
    p_wide = Parameter(
        label="toy.y", initval=-1.0, lower=-5.0, upper=5.0, init_scale=0.3
    )
    p_gauss = Parameter(
        label="toy.z",
        initval=0.5,
        mu=0.0,
        sigma=2.0,
        lower=-np.inf,
        upper=np.inf,
    )
    with pm.Model() as model:
        xv = p_tight.build_pymc()
        yv = p_wide.build_pymc()
        zv = p_gauss.build_pymc()
        pm.Potential(
            "like",
            -0.5 * ((xv - 2.5) / 0.01) ** 2 - 0.5 * ((yv + 1.2) / 2.0) ** 2,
        )
    return model, p_tight, p_wide, p_gauss


class _StubSystem:
    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params

    def get_raw_start(self, model):
        return model.initial_point()


def test_posterior_invariant_under_set_whitening():
    """Given a bounded sampled parameter,
    When set_whitening rescales it by orders of magnitude,
    Then the logp of the SAME PHYSICAL point is bit-for-bit-close and the
    same compiled logp function sees the new scales without recompiling.
    """
    # Arrange
    model, p_tight, p_wide, _ = _toy_model()
    logp = model.compile_logp()
    det = model.compile_fn(
        model.replace_rvs_by_values([model["toy.x"], model["toy.y"]])
    )
    raw_pt = {
        "toy.x_raw": np.array([0.7]),
        "toy.y_raw": np.array([-0.3]),
        "toy.z_raw": np.array([0.2]),
    }
    det_pt = {k: v for k, v in raw_pt.items() if k != "toy.z_raw"}
    x1, y1 = det(det_pt)
    lp_before = float(logp(raw_pt))

    # Act
    p_tight.set_whitening(np.array([0.01]))
    p_wide.set_whitening(np.array([37.0]))

    # Assert: recover the same physical point in the new raw coordinates
    raw_pt2 = {
        "toy.x_raw": p_tight.raw_from_initval(np.array([x1])),
        "toy.y_raw": p_wide.raw_from_initval(np.array([y1])),
        "toy.z_raw": np.array([0.2]),
    }
    x2, y2 = det({k: v for k, v in raw_pt2.items() if k != "toy.z_raw"})
    assert np.isclose(x2, x1, rtol=1e-9)
    assert np.isclose(y2, y1, rtol=1e-9)
    assert np.isclose(float(logp(raw_pt2)), lp_before, rtol=1e-8, atol=1e-8)


def test_raw_zero_still_maps_to_initval_after_rescale():
    """Given a rescale by a large factor in either direction,
    When the model is evaluated at raw = 0,
    Then the physical value is still exactly the initval (the start the
    probe measured around must not move).
    """
    # Arrange
    model, p_tight, p_wide, _ = _toy_model()
    det = model.compile_fn(
        model.replace_rvs_by_values([model["toy.x"], model["toy.y"]])
    )
    ip = model.initial_point()

    # Act
    p_tight.set_whitening(np.array([1e-4]))
    p_wide.set_whitening(np.array([50.0]))

    # Assert
    x0, y0 = det({k: v for k, v in ip.items() if k != "toy.z_raw"})
    assert np.isclose(x0, 2.0)
    assert np.isclose(y0, -1.0)


def test_gaussian_prior_scale_is_never_touched():
    """Given an unbounded parameter whose raw N(0,1) IS the prior,
    When set_whitening is called on it,
    Then its transform is unchanged (rescaling it would change the
    posterior, not just the conditioning).
    """
    # Arrange
    model, _, _, p_gauss = _toy_model()
    tf_before = p_gauss._raw_transform["gaussian_scales"].copy()

    # Act
    p_gauss.set_whitening(np.array([100.0]))

    # Assert
    np.testing.assert_array_equal(
        p_gauss._raw_transform["gaussian_scales"], tf_before
    )


def test_nonfinite_multiplier_keeps_preliminary_scale():
    """Given a flat probe direction (NaN multiplier, whitening's flat marker),
    When set_whitening is applied,
    Then that element's scale is unchanged instead of poisoned.
    """
    # Arrange
    model, p_tight, _, _ = _toy_model()
    before = p_tight._whiten_state["sv_scale_logits"].get_value().copy()

    # Act
    p_tight.set_whitening(np.array([np.nan]))

    # Assert
    np.testing.assert_array_equal(
        p_tight._whiten_state["sv_scale_logits"].get_value(), before
    )


def test_apply_measured_whitening_reprobe_lands_at_unity():
    """Given a model whose preliminary scales are off by orders of magnitude,
    When apply_measured_whitening probes and rescales it,
    Then a re-probe of the SAME model measures ~1 for every constrained
    element (within the probe's 5% bisection tolerance), i.e. one raw unit
    now costs ~0.5 nats everywhere -- the conditioning the retired
    curvature check asked users to hand-tune toward.
    """
    # Arrange
    model, p_tight, p_wide, p_gauss = _toy_model()
    system = _StubSystem([p_tight, p_wide, p_gauss])
    logp = model.compile_logp()
    raw_start = model.initial_point()

    # Act
    report = apply_measured_whitening(system, model, raw_start, logp_fn=logp)

    # Assert: report carries a multiplier per raw element
    assert set(report["multipliers"]) == {
        "toy.x_raw",
        "toy.y_raw",
        "toy.z_raw",
    }
    _, scales2 = probe_scales(raw_start, logp)
    for key in ("toy.x_raw", "toy.y_raw"):
        np.testing.assert_allclose(scales2[key], 1.0, rtol=0.12, err_msg=key)
    # The Gaussian-prior element is deliberately NOT rescaled (its raw
    # N(0,1) IS the prior), so its re-probe matches the original probe:
    # the nearer 0.5-nat contour from the off-mode start raw=0.25 is
    # sqrt(0.25^2 + 1) - 0.25 ~ 0.78, not 1.
    np.testing.assert_allclose(
        scales2["toy.z_raw"], report["multipliers"]["toy.z_raw"], rtol=0.12
    )


def test_probe_is_deterministic():
    """Given the same model and start,
    When the probe runs twice,
    Then the measured scales are identical (trace reloads that rebuild the
    model re-derive the exact same whitening).
    """
    # Arrange
    model, *_ = _toy_model()
    logp = model.compile_logp()
    raw_start = model.initial_point()

    # Act
    _, s1 = probe_scales(raw_start, logp)
    _, s2 = probe_scales(raw_start, logp)

    # Assert
    for k in s1:
        np.testing.assert_array_equal(s1[k], s2[k])


def test_user_init_scale_is_stripped_with_warning(caplog):
    """Given a params dict containing init_scale entries,
    When a ConfigManager is constructed,
    Then the keys are removed and a warning names the offending paths.
    """
    # Arrange
    import logging

    from exozippy.config import ConfigManager

    user_params = {
        "star.0.mass": {"initval": 1.0, "init_scale": 0.1},
        "star.0.radius": {"initval": 1.0},
    }

    # Act
    with caplog.at_level(logging.WARNING):
        cm = ConfigManager(user_params)

    # Assert
    assert "init_scale" not in cm.user_params["star.0.mass"]
    assert cm.user_params["star.0.mass"]["initval"] == 1.0
    assert any(
        "obsolete" in rec.message and "star.0.mass" in rec.message
        for rec in caplog.records
    )


def test_missing_init_scale_falls_back_to_span_fraction():
    """Given a bounded sampled parameter with no init_scale anywhere,
    When the model is built,
    Then the preliminary whitening scale is a fraction of the span (the
    parameter builds and samples rather than erroring)."""
    # Arrange / Act
    p = Parameter(label="toy.a", initval=5.0, lower=0.0, upper=100.0)
    with pm.Model() as model:
        p.build_pymc()

    # Assert: raw=0 -> initval, and one raw unit moves a span-fraction step
    det = model.compile_fn(model.replace_rvs_by_values([model["toy.a"]]))
    (a0,) = det(model.initial_point())
    assert np.isclose(a0, 5.0)
    (a1,) = det({"toy.a_raw": np.array([1.0])})
    step = a1 - a0
    assert 1.0 < step < 20.0  # ~ 0.1 * span, through the logit Jacobian


# ---------------------------------------------------------------------------
# Round 2: raw_scales for PTDE, escalation, barrier scales, persistence
# ---------------------------------------------------------------------------


def test_report_raw_scales_for_ptde():
    """Given the whitening rescale,
    When the report is built,
    Then raw_scales carries the post-rescale per-element dispersion PTDE
    uses instead of re-probing: exactly 1.0 for rescaled elements, the
    measured value for the untouched Gaussian-prior element.
    """
    # Arrange
    model, p_tight, p_wide, p_gauss = _toy_model()
    system = _StubSystem([p_tight, p_wide, p_gauss])

    # Act
    report = apply_measured_whitening(system, model)

    # Assert
    np.testing.assert_array_equal(report["raw_scales"]["toy.x_raw"], 1.0)
    np.testing.assert_array_equal(report["raw_scales"]["toy.y_raw"], 1.0)
    z = float(np.asarray(report["raw_scales"]["toy.z_raw"])[0])
    assert 0.5 < z < 1.0  # nearer contour from the off-mode start, not 1


def test_make_starts_skips_probe_when_raw_scales_given(monkeypatch):
    """Given precomputed raw_scales from the whitening pass,
    When PTDE builds its chain starts,
    Then the probe is never called (the model was whitened against the
    same start) and the provided scales disperse the chains.
    """
    # Arrange -- patch the probe where _make_starts actually resolves it
    # (samplers._common, the shared scaffolding module; ptde re-exports it).
    from exozippy.samplers import _common, ptde

    def _boom(*a, **k):
        raise AssertionError("probe should not run")

    monkeypatch.setattr(_common, "_probe_scales", _boom)
    rng = np.random.default_rng(0)
    start = {"a": np.zeros(2)}

    def logp_fn(pt_):
        return -0.5 * float(np.sum(pt_["a"] ** 2))

    # Act
    starts, seed_idx = ptde._make_starts(
        4, start, logp_fn, rng, raw_scales={"a": np.ones(2)}
    )

    # Assert
    assert len(starts) == 4
    assert seed_idx == [0, 0, 0, 0]


def test_escalation_resolves_scales_beyond_probe_range():
    """Given a parameter whose true scale is below the probe's single-round
    floor (preliminary scale off by ~14 orders of magnitude, e.g. a period
    constrained to nanoseconds against day-scale bounds),
    When apply_measured_whitening runs,
    Then the escalation rounds resolve it: a re-probe of the rescaled model
    lands near 1 instead of the clipped floor.
    """
    # Arrange: true posterior sigma ~5e-15 of the ~unit preliminary scale
    p = Parameter(label="toy.t", initval=2.0, lower=0.0, upper=10.0)
    with pm.Model() as model:
        tv = p.build_pymc()
        pm.Potential("like", -0.5 * ((tv - 2.0) / 5.0e-15) ** 2)
    system = _StubSystem([p])
    logp = model.compile_logp()
    raw_start = model.initial_point()

    # Act
    apply_measured_whitening(system, model, raw_start, logp_fn=logp)

    # Assert
    _, scales2 = probe_scales(raw_start, logp)
    final = float(np.asarray(scales2["toy.t_raw"])[0])
    assert 0.3 < final < 3.0, final


def _too_tight_model(init_scale, true_sigma=1.0):
    """A bounded parameter whose preliminary scale is far too TIGHT.

    The mirror of the scenario above: there the preliminary scale was too
    loose (the probe hits its step FLOOR), here it is too tight, so the probe
    must walk far out in raw units -- past parameter.py's raw-cancellation
    clip, which is the wall review 1.2.1 is about.
    """
    p = Parameter(
        label="toy.t",
        initval=2.0,
        lower=0.0,
        upper=10.0,
        init_scale=init_scale,
    )
    with pm.Model() as model:
        tv = p.build_pymc()
        pm.Potential("like", -0.5 * ((tv - 2.0) / true_sigma) ** 2)
    return model, p


def test_escalation_resolves_a_far_too_tight_preliminary_scale(caplog):
    """Given a preliminary scale 10 orders of magnitude too TIGHT
    (init_scale 1e-10 on [0, 10] against a true sigma of 1.0 -- review
    1.2.1's own reproduction),
    When apply_measured_whitening runs,
    Then the escalation resolves it -- a re-probe of the rescaled model lands
    near 1 -- and no "still unresolved" warning is emitted.

    Before the fix the measured multiplier was exactly the raw-cancellation
    clip (1e4): that wall manufactures a 0.5-nat contour the probe cannot
    see past, it sat well INSIDE the escalation window, so no round ran, no
    warning fired, and the model was left ~1e6 under-whitened.
    """
    import logging

    # Arrange
    model, p = _too_tight_model(1.0e-10)
    system = _StubSystem([p])
    logp = model.compile_logp()
    raw_start = model.initial_point()

    # Act
    with caplog.at_level(logging.WARNING, logger="exozippy.whitening"):
        report = apply_measured_whitening(
            system, model, raw_start, logp_fn=logp
        )

    # Assert: the cumulative correction is the ~1e10 the setup asks for, not
    # the 1e4 wall, and the rescaled model probes at ~1.
    cumulative = float(np.asarray(report["multipliers"]["toy.t_raw"])[0])
    assert 1e9 < cumulative < 1e11, cumulative
    _, scales2 = probe_scales(raw_start, logp)
    final = float(np.asarray(scales2["toy.t_raw"])[0])
    assert 0.3 < final < 3.0, final
    assert "still unresolved" not in caplog.text


def test_probe_raises_the_raw_cancellation_clip_and_restores_it():
    """Given the raw-cancellation clip is a sampler safety device,
    When the whitening probe measures a too-tight element,
    Then the clip is raised for the measurement and restored afterwards --
    sampling must never run at the probe's value.
    """
    from exozippy.components import parameter as parameter_mod
    from exozippy.whitening import _PROBE_RAW_CLIP

    # Arrange
    model, p = _too_tight_model(1.0e-10)
    system = _StubSystem([p])
    inner_logp = model.compile_logp()
    seen = []

    def logp(point):
        seen.append(parameter_mod.get_raw_cancellation_clip())
        return inner_logp(point)

    before = parameter_mod.get_raw_cancellation_clip()

    # Act
    apply_measured_whitening(system, model, model.initial_point(), logp)

    # Assert
    assert before == parameter_mod._RAW_CANCELLATION_CLIP
    assert seen and set(seen) == {_PROBE_RAW_CLIP}
    assert parameter_mod.get_raw_cancellation_clip() == before


def test_escalation_iterates_past_two_rounds(caplog):
    """Given a preliminary scale 19 orders of magnitude too tight -- past the
    two rounds the escalation used to be hardcoded to (~6.5 orders each),
    When apply_measured_whitening runs,
    Then it keeps escalating until nothing is clipped: THREE rounds run and
    the scale still resolves, so the reach is not a hardcoded number.
    """
    import logging

    # Arrange
    model, p = _too_tight_model(1.0e-19)
    system = _StubSystem([p])
    logp = model.compile_logp()
    raw_start = model.initial_point()

    # Act
    with caplog.at_level(logging.WARNING, logger="exozippy.whitening"):
        apply_measured_whitening(system, model, raw_start, logp_fn=logp)

    # Assert
    rounds = caplog.text.count("escalation round")
    assert rounds >= 3, caplog.text
    _, scales2 = probe_scales(raw_start, logp)
    final = float(np.asarray(scales2["toy.t_raw"])[0])
    assert 0.3 < final < 3.0, final
    assert "still unresolved" not in caplog.text


def test_no_false_still_unresolved_warning_after_escalation(caplog):
    """Given the too-LOOSE scenario the shipped escalation test uses
    (true sigma 5e-15, resolved by one round to a cumulative ~5e-15),
    When apply_measured_whitening finishes,
    Then no "still unresolved" warning is emitted: the test is on the LATEST
    measured multiplier, not on the cumulative one -- which is outside the
    window by construction after any successful escalation (review 1.2.2).
    """
    import logging

    # Arrange
    p = Parameter(label="toy.t", initval=2.0, lower=0.0, upper=10.0)
    with pm.Model() as model:
        tv = p.build_pymc()
        pm.Potential("like", -0.5 * ((tv - 2.0) / 5.0e-15) ** 2)
    system = _StubSystem([p])
    logp = model.compile_logp()

    # Act
    with caplog.at_level(logging.WARNING, logger="exozippy.whitening"):
        report = apply_measured_whitening(
            system, model, model.initial_point(), logp_fn=logp
        )

    # Assert
    cumulative = float(np.asarray(report["multipliers"]["toy.t_raw"])[0])
    assert cumulative < _CLIP_LO  # the cumulative IS outside the window
    assert "still unresolved" not in caplog.text


def _barrier_model():
    """x, y sampled; d = x + y derived with finite bounds -> soft barriers."""
    import pytensor.tensor as pt

    p_x = Parameter(label="toy.x", initval=2.0, lower=0.0, upper=10.0)
    p_y = Parameter(
        label="toy.y", initval=-1.0, lower=-5.0, upper=5.0, init_scale=0.3
    )
    holder = {}
    p_d = Parameter(
        label="toy.d",
        lower=-50.0,
        upper=50.0,
        expression=lambda: holder["x"] + holder["y"],
    )
    with pm.Model() as model:
        holder["x"] = p_x.build_pymc()
        holder["y"] = p_y.build_pymc()
        p_d.build_pymc()
        pm.Potential(
            "like",
            -0.5 * ((holder["x"] - 2.5) / 0.01) ** 2
            - 0.5 * ((holder["y"] + 1.2) / 2.0) ** 2,
        )
    return model, p_x, p_y, p_d


def test_barrier_scales_measured_from_unit_step_response():
    """Given a derived bounded parameter d = x + y,
    When the model is whitened and measure_barrier_scales runs,
    Then d's shared barrier scale becomes the quadrature sum of its actual
    responses to unit raw steps -- measured, not sympy-propagated.
    """
    from exozippy.whitening import measure_barrier_scales

    # Arrange
    model, p_x, p_y, p_d = _barrier_model()
    system = _StubSystem([p_x, p_y, p_d])
    raw_start = model.initial_point()
    assert p_d._barrier_state is not None
    prelim = p_d._barrier_state["sv"].get_value().copy()

    apply_measured_whitening(system, model, raw_start)

    # Independent expectation: evaluate d at the start and at unit steps.
    # (d has no sampled elements of its own, so it is not a named
    # Deterministic -- reach it through the Parameter's value node.)
    det = model.compile_fn(
        model.replace_rvs_by_values([p_d.value]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    d0 = float(np.asarray(det(raw_start)[0]))
    resp = []
    for key in ("toy.x_raw", "toy.y_raw"):
        stepped = {k: v.copy() for k, v in raw_start.items()}
        stepped[key] = stepped[key] + 1.0
        resp.append(float(np.asarray(det(stepped)[0])) - d0)
    expected = np.sqrt(np.sum(np.square(resp)))

    # Act
    measured = measure_barrier_scales(system, model, raw_start)

    # Assert
    import pytest

    got = float(p_d._barrier_state["sv"].get_value()[0])
    assert got == pytest.approx(expected, rel=1e-9)
    assert got != prelim[0]
    assert "toy.d" in measured


def test_user_bound_scale_pins_the_barrier():
    """Given a user bound_scale on the derived parameter,
    When measure_barrier_scales runs,
    Then the pinned element keeps the user's steepness scale.
    """
    # Arrange
    import pytensor.tensor as pt

    from exozippy.whitening import measure_barrier_scales

    p_x = Parameter(label="toy.x", initval=2.0, lower=0.0, upper=10.0)
    holder = {}
    p_d = Parameter(
        label="toy.d",
        lower=-50.0,
        upper=50.0,
        bound_scale=7.5,
        expression=lambda: holder["x"] * 2.0,
    )
    with pm.Model() as model:
        holder["x"] = p_x.build_pymc()
        p_d.build_pymc()
        pm.Potential("like", -0.5 * ((holder["x"] - 2.5) / 0.01) ** 2)
    system = _StubSystem([p_x, p_d])
    raw_start = model.initial_point()
    apply_measured_whitening(system, model, raw_start)

    # Act
    measure_barrier_scales(system, model, raw_start)

    # Assert
    assert float(p_d._barrier_state["sv"].get_value()[0]) == 7.5


def test_whitening_persistence_round_trip(tmp_path):
    """Given a measured-and-whitened model,
    When the state is saved and loaded into the same build,
    Then load succeeds and the shared values are identical; a state that no
    longer matches the model is rejected without touching anything.
    """
    from exozippy.whitening import (
        load_whitening,
        measure_and_whiten,
        save_whitening,
    )

    # Arrange
    model, p_x, p_y, p_d = _barrier_model()
    system = _StubSystem([p_x, p_y, p_d])
    raw_start = model.initial_point()
    report = measure_and_whiten(system, model, raw_start)
    path = tmp_path / "wh.json"
    save_whitening(system, str(path), map_lp=report["map_lp"])
    before = p_x._whiten_state["sv_scale_logits"].get_value().copy()

    # Act / Assert: round trip
    assert load_whitening(system, str(path))
    np.testing.assert_array_equal(
        p_x._whiten_state["sv_scale_logits"].get_value(), before
    )

    # Act / Assert: a mismatched state (missing coverage) is rejected and
    # leaves the build untouched.
    import json

    data = json.loads(path.read_text())
    del data["params"]["toy.x"]
    path.write_text(json.dumps(data))
    assert not load_whitening(system, str(path))
    np.testing.assert_array_equal(
        p_x._whiten_state["sv_scale_logits"].get_value(), before
    )


# ---------------------------------------------------------------------------
# Review 2.10.1 / 2.10.4
# ---------------------------------------------------------------------------


def test_unbounded_no_sigma_posterior_invariant_under_rescale():
    """Given a sampled element with INFINITE bounds and no sigma -- so
    nothing cancels its raw N(0,1) and that raw prior IS its prior,
    When set_whitening applies a probe-measured multiplier,
    Then the logp as a function of the PHYSICAL value is unchanged.

    Review 2.10.1: the rescale used to multiply this element's linear scale,
    turning its prior from N(initval, init_scale) into N(initval,
    m*init_scale) -- and m is measured from the data, so the prior width
    became data-dependent.  The neighbouring unbounded-WITH-sigma case was
    already excluded for exactly this reason; the guard was written as "has
    an explicit sigma" instead of "the raw N(0,1) is the prior".
    """
    # Arrange
    p = Parameter(
        label="toy.u",
        initval=0.0,
        init_scale=1.0,
        lower=-np.inf,
        upper=np.inf,
    )
    with pm.Model() as model:
        p.build_pymc()
    logp = model.compile_logp()

    def lp_at_physical(v):
        gs = p._whiten_state["sv_gaussian_scales"].get_value()[0]
        mu = p._raw_transform["gaussian_mus"][0]
        return float(logp({"toy.u_raw": np.array([(v - mu) / gs])}))

    physical = [0.0, 1.0, 2.0]
    before = np.array([lp_at_physical(v) for v in physical])

    # Act
    p.set_whitening(np.array([3.0]))

    # Assert -- same density over physical space, up to a constant
    after = np.array([lp_at_physical(v) for v in physical])
    np.testing.assert_allclose(
        after - after[0], before - before[0], rtol=1e-10, atol=1e-10
    )


def test_unbounded_no_sigma_scale_is_reported_not_applied():
    """Given the same element,
    When set_whitening runs,
    Then the shared linear scale is untouched and the measured value comes
    back in the returned post-rescale scales (what PTDE disperses chains by).
    """
    # Arrange
    p = Parameter(
        label="toy.u",
        initval=0.0,
        init_scale=1.0,
        lower=-np.inf,
        upper=np.inf,
    )
    with pm.Model():
        p.build_pymc()
    before = p._whiten_state["sv_gaussian_scales"].get_value().copy()

    # Act
    post = p.set_whitening(np.array([7.0]))

    # Assert
    np.testing.assert_array_equal(
        p._whiten_state["sv_gaussian_scales"].get_value(), before
    )
    np.testing.assert_array_equal(p._raw_transform["gaussian_scales"], before)
    assert post.tolist() == [7.0]


def test_load_whitening_rejects_a_file_missing_a_barrier_only_parameter(
    tmp_path,
):
    """Given a persisted state that omits a DERIVED, barrier-only parameter
    (no _whiten_state, so the old coverage check never looked at it),
    When it is loaded,
    Then the load is rejected and the caller re-measures.

    Review 2.10.4: barriers ARE posterior terms, so a file that does not
    describe every barrier describes a different logp than the one that
    produced the trace being reused.
    """
    import json

    from exozippy.whitening import (
        load_whitening,
        measure_and_whiten,
        save_whitening,
    )

    # Arrange
    model, p_x, p_y, p_d = _barrier_model()
    system = _StubSystem([p_x, p_y, p_d])
    raw_start = model.initial_point()
    report = measure_and_whiten(system, model, raw_start)
    path = tmp_path / "wh.json"
    save_whitening(system, str(path), map_lp=report["map_lp"])

    assert p_d._whiten_state is None and p_d._barrier_state is not None
    data = json.loads(path.read_text())
    assert "toy.d" in data["params"]
    del data["params"]["toy.d"]
    path.write_text(json.dumps(data))
    barrier_before = p_d._barrier_state["sv"].get_value().copy()

    # Act
    ok = load_whitening(system, str(path))

    # Assert
    assert ok is False
    np.testing.assert_array_equal(
        p_d._barrier_state["sv"].get_value(), barrier_before
    )


def test_load_whitening_never_applies_a_state_partially(tmp_path):
    """Given a persisted state whose FIRST parameter is valid and whose
    second carries a mismatched vector,
    When it is loaded,
    Then nothing at all is applied -- the earlier parameter keeps its own
    scales instead of being left half-restored.

    Review 2.10.4: only scale_logits was validated up front, so a bad
    gaussian_scales aborted inside the apply loop, after earlier parameters
    had already been written, and the function still returned True.
    """
    import json

    from exozippy.whitening import (
        load_whitening,
        measure_and_whiten,
        save_whitening,
    )

    # Arrange
    model, p_x, p_y, p_d = _barrier_model()
    system = _StubSystem([p_x, p_y, p_d])
    raw_start = model.initial_point()
    report = measure_and_whiten(system, model, raw_start)
    path = tmp_path / "wh.json"
    save_whitening(system, str(path), map_lp=report["map_lp"])

    data = json.loads(path.read_text())
    # A value that would be visible if it were applied.
    data["params"]["toy.x"]["scale_logits"] = [123.0]
    # ...and a second parameter the apply loop would choke on.
    data["params"]["toy.y"]["gaussian_scales"] = [1.0, 2.0]
    path.write_text(json.dumps(data))
    x_before = p_x._whiten_state["sv_scale_logits"].get_value().copy()

    # Act
    ok = load_whitening(system, str(path))

    # Assert
    assert ok is False
    np.testing.assert_array_equal(
        p_x._whiten_state["sv_scale_logits"].get_value(), x_before
    )


# ---------------------------------------------------------------------------
# Reuse path: the whitening a saved trace was sampled under is a property of
# draws that already exist, not a free choice of coordinates.  It is restored,
# never re-measured, and never rewritten.
# ---------------------------------------------------------------------------


def _saved_whitening_setup(tmp_path):
    """A whitened toy build plus the file its state was persisted to."""
    from exozippy.whitening import measure_and_whiten, save_whitening

    model, p_x, p_y, p_d = _barrier_model()
    system = _StubSystem([p_x, p_y, p_d])
    raw_start = model.initial_point()
    report = measure_and_whiten(system, model, raw_start)
    path = tmp_path / "fit_whitening.json"
    save_whitening(system, str(path), map_lp=report["map_lp"])
    return model, system, (p_x, p_y, p_d), path


def test_restore_for_trace_applies_a_matching_file_without_rewriting_it(
    tmp_path,
):
    """Given a whitening file that matches the build,
    When it is restored on the trace-reuse path,
    Then it applies and the file on disk is left byte-for-byte alone.
    """
    from exozippy.whitening import restore_whitening_for_trace

    # Arrange
    _model, system, (p_x, _p_y, _p_d), path = _saved_whitening_setup(tmp_path)
    before_bytes = path.read_bytes()
    before_scales = p_x._whiten_state["sv_scale_logits"].get_value().copy()

    # Act
    status = restore_whitening_for_trace(system, str(path), "fit_trace.nc")

    # Assert
    assert status == "restored"
    assert path.read_bytes() == before_bytes
    np.testing.assert_array_equal(
        p_x._whiten_state["sv_scale_logits"].get_value(), before_scales
    )


def test_restore_for_trace_raises_instead_of_silently_recoordinating(
    tmp_path,
):
    """Given a saved trace being REUSED and a whitening file that no longer
    describes the build (it omits a barrier-only parameter),
    When the whitening is restored,
    Then StaleWhiteningError is raised naming the file, the reason and the
    remedy -- and the file is NOT overwritten.

    The gap this closes: run.py used to fall back to
    measure_and_whiten + save_whitening here.  The trace's raw draws are
    coordinates in the whitened space, so re-probing silently changed what
    every stored draw decodes to -- and the overwrite destroyed the only
    record of the coordinates they were actually sampled in.
    """
    import json

    import pytest

    from exozippy.whitening import (
        StaleWhiteningError,
        restore_whitening_for_trace,
    )

    # Arrange
    _model, system, (p_x, _p_y, p_d), path = _saved_whitening_setup(tmp_path)
    data = json.loads(path.read_text())
    assert p_d._whiten_state is None and p_d._barrier_state is not None
    del data["params"]["toy.d"]
    path.write_text(json.dumps(data))
    before_bytes = path.read_bytes()
    x_before = p_x._whiten_state["sv_scale_logits"].get_value().copy()
    d_before = p_d._barrier_state["sv"].get_value().copy()

    # Act
    with pytest.raises(StaleWhiteningError) as excinfo:
        restore_whitening_for_trace(system, str(path), "fit_trace.nc")

    # Assert -- diagnosable
    message = str(excinfo.value)
    assert str(path) in message
    assert "fit_trace.nc" in message
    assert "toy.d" in message  # WHY it does not apply
    assert "recompute_trace: true" in message  # the remedy
    # ...and nothing was rewritten or re-coordinated.
    assert path.read_bytes() == before_bytes
    np.testing.assert_array_equal(
        p_x._whiten_state["sv_scale_logits"].get_value(), x_before
    )
    np.testing.assert_array_equal(
        p_d._barrier_state["sv"].get_value(), d_before
    )


def test_restore_for_trace_warns_and_keeps_preliminary_scales_when_absent(
    tmp_path, caplog
):
    """Given a reused trace with NO whitening file beside it,
    When the whitening is restored,
    Then it is reported as unverifiable (not stale), the build keeps its
    PRELIMINARY scales -- nothing is probed -- and no file is written.

    Absent state is the trace_meta "unverifiable fingerprint" case, not the
    stale one: a trace sampled with 'measure_scales: false' (or before the
    state was persisted at all) legitimately has no file, and for the former
    the preliminary scales ARE the ones it was sampled with.  Writing a
    freshly measured file here would be worse than useless: the NEXT reload
    would restore it silently, with no warning left to notice.
    """
    import logging

    from exozippy.whitening import restore_whitening_for_trace

    # Arrange
    model, p_x, p_y, p_d = _barrier_model()
    system = _StubSystem([p_x, p_y, p_d])
    prelim_x = p_x._whiten_state["sv_scale_logits"].get_value().copy()
    prelim_d = p_d._barrier_state["sv"].get_value().copy()
    missing = tmp_path / "fit_whitening.json"

    # Act
    with caplog.at_level(logging.WARNING):
        status = restore_whitening_for_trace(
            system, str(missing), "fit_trace.nc"
        )

    # Assert
    assert status == "unverifiable"
    assert not missing.exists()
    np.testing.assert_array_equal(
        p_x._whiten_state["sv_scale_logits"].get_value(), prelim_x
    )
    np.testing.assert_array_equal(
        p_d._barrier_state["sv"].get_value(), prelim_d
    )
    text = caplog.text
    assert "UNVERIFIABLE WHITENING" in text
    assert str(missing) in text
    assert "fit_trace.nc" in text
    assert "recompute_trace: true" in text


def test_fresh_path_still_falls_back_and_re_measures(tmp_path, caplog):
    """Given the same invalid whitening file but a run that is about to
    SAMPLE (no trace being reused),
    When load_whitening is called,
    Then it still warns and returns False so the caller re-measures and
    overwrites -- the coordinates are a free choice until draws exist, and
    that path must not regress into raising.
    """
    import json
    import logging

    from exozippy.whitening import load_whitening

    # Arrange
    _model, system, (_p_x, _p_y, p_d), path = _saved_whitening_setup(tmp_path)
    data = json.loads(path.read_text())
    del data["params"]["toy.d"]
    path.write_text(json.dumps(data))

    # Act
    with caplog.at_level(logging.WARNING):
        ok = load_whitening(system, str(path))

    # Assert
    assert ok is False
    assert "re-measuring" in caplog.text


# ---------------------------------------------------------------------------
# prepare_whitening: the branch run.py actually takes.  The bug was that both
# arms of it did the same thing.
# ---------------------------------------------------------------------------


def _break_the_file(path):
    """Drop the barrier-only parameter, so validation rejects the file."""
    import json

    data = json.loads(path.read_text())
    del data["params"]["toy.d"]
    path.write_text(json.dumps(data))


def test_prepare_whitening_fresh_run_still_remeasures_and_overwrites(
    tmp_path,
):
    """Given an invalid whitening file and a run that is about to SAMPLE,
    When prepare_whitening runs with reusing_trace=False,
    Then it measures and OVERWRITES the file, exactly as before.

    This path is correct: no draws exist yet, so the coordinates are still
    a free choice and a stale file must not block a fresh fit.  It must not
    regress into the reuse path's raise.
    """
    from exozippy.whitening import prepare_whitening

    # Arrange
    model, system, (_p_x, _p_y, p_d), path = _saved_whitening_setup(tmp_path)
    _break_the_file(path)
    stale_bytes = path.read_bytes()

    # Act
    report = prepare_whitening(
        system,
        model,
        model.initial_point(),
        str(path),
        str(tmp_path / "fit_trace.nc"),
        reusing_trace=False,
    )

    # Assert
    assert report is not None and "map_lp" in report
    assert path.read_bytes() != stale_bytes
    import json

    assert "toy.d" in json.loads(path.read_text())["params"]


def test_prepare_whitening_reuse_path_raises_and_leaves_the_file(tmp_path):
    """Given the SAME invalid whitening file but a saved trace being reused,
    When prepare_whitening runs with reusing_trace=True,
    Then it raises StaleWhiteningError and the file is untouched.

    Pre-fix, run.py took the fresh branch here too: the reused trace's raw
    draws silently began decoding under a whitening they were never sampled
    with, and the file that said what they WERE sampled with was overwritten
    in the same breath.
    """
    import pytest

    from exozippy.whitening import StaleWhiteningError, prepare_whitening

    # Arrange
    model, system, _params, path = _saved_whitening_setup(tmp_path)
    _break_the_file(path)
    stale_bytes = path.read_bytes()

    # Act / Assert
    with pytest.raises(StaleWhiteningError):
        prepare_whitening(
            system,
            model,
            model.initial_point(),
            str(path),
            str(tmp_path / "fit_trace.nc"),
            reusing_trace=True,
        )
    assert path.read_bytes() == stale_bytes


def test_prepare_whitening_reuse_path_writes_nothing_when_absent(tmp_path):
    """Given a reused trace with no whitening file beside it,
    When prepare_whitening runs with reusing_trace=True,
    Then no probe is run and no file is created -- so the NEXT reload warns
    again instead of silently trusting a file the trace never saw.
    """
    from exozippy.whitening import prepare_whitening

    # Arrange
    model, p_x, p_y, p_d = _barrier_model()
    system = _StubSystem([p_x, p_y, p_d])
    prelim = p_x._whiten_state["sv_scale_logits"].get_value().copy()
    path = tmp_path / "fit_whitening.json"

    # Act
    report = prepare_whitening(
        system,
        model,
        model.initial_point(),
        str(path),
        str(tmp_path / "fit_trace.nc"),
        reusing_trace=True,
    )

    # Assert
    assert report is None
    assert not path.exists()
    np.testing.assert_array_equal(
        p_x._whiten_state["sv_scale_logits"].get_value(), prelim
    )
