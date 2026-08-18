"""Mass-ratio (q) sanitization and the alpha_deg consolidation.

Review item 4.5: ``clip(nan_to_num(q), 1e-9, 100)`` was copied to five sites
and ``arctan2(yalpha, xalpha) * 180/pi`` to two.  Both now have exactly one
implementation -- ``physics.clip_q`` / ``physics.clip_q_value`` and
``Lens._alpha_deg`` -- and the ``nan_to_num`` half is gone, because it could
only ever invent a mass ratio for a computation that had already failed.
"""

import inspect
import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.mulensing import lens as lens_mod
from exozippy.components.mulensing.lens import Lens
from exozippy.components.mulensing.op import (
    BinaryLensMagOp,
    VBMDirectMagOp,
    _build_binary_model,
)
from exozippy.components.mulensing.physics import (
    Q_MAX,
    Q_MIN,
    calc_alpha,
    clip_q,
    clip_q_value,
)
from exozippy.system import System

_COORDS = "268.0d -29.0d"


def _binary_config():
    """A minimal 2L1S topology: star.0 + planet.0 lens, star.1 source."""
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "Companion"}],
        "lens": [
            {
                "name": "Lens",
                "lenses": ["star.0", "planet.0"],
                "sources": ["star.1"],
                "finite_source": False,
            }
        ],
    }
    user_params = {
        "lens.Lens.t_0": {"initval": 2458554.89},
        "lens.Lens.u_0": {"initval": 0.1},
        "lens.Lens.t_E": {"initval": 18.2},
        "lens.Lens.s": {"initval": 0.98},
        "lens.Lens.alpha": {"initval": -52.0},
        "lens.Lens.q": {"initval": 1.1e-3},
        "star.Lens.ra": {"initval": 268.0, "sigma": 0},
        "star.Lens.dec": {"initval": -29.0, "sigma": 0},
        "star.Source.ra": {"initval": 268.0, "sigma": 0},
        "star.Source.dec": {"initval": -29.0, "sigma": 0},
    }
    return config, user_params


@pytest.fixture(scope="module")
def binary_system():
    config, user_params = _binary_config()
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    return system, model


# ---------------------------------------------------------------------------
# The consolidated helper: in-range behaviour is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [Q_MIN, 1e-8, 1.1e-3, 0.5, 1.0, 1.188, 99.0, Q_MAX],
)
def test_clip_q_is_the_identity_inside_the_valid_range(value):
    """
    Given a finite mass ratio inside [Q_MIN, Q_MAX],
    When clip_q (symbolic) or clip_q_value (numeric) is applied,
    Then the value passes through bit-identically -- the consolidation must
      not perturb any working fit.
    """
    assert float(clip_q(pt.as_tensor_variable(value)).eval()) == value
    assert clip_q_value(value) == value


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, Q_MIN),
        (-1.0, Q_MIN),
        (1e-30, Q_MIN),
        (1e5, Q_MAX),
        (np.inf, Q_MAX),
        (-np.inf, Q_MIN),
    ],
)
def test_clip_q_clamps_out_of_range_but_finite_input(value, expected):
    """
    Given a mass ratio outside the range the magnification backends are
      defined on (including the infinities, which the old nan_to_num mapped
      to +/-float64.max before the clip caught them),
    When the helper is applied,
    Then it is clamped to the bound -- unchanged from the old expression, on
      both the symbolic and the numeric path.  An infinity carries a sign, so
      it is an ordinary out-of-range value and clipping it is the same
      modelling decision; NaN (below) has no value at all and is not.
    """
    assert float(clip_q(pt.as_tensor_variable(value)).eval()) == expected
    assert clip_q_value(value) == expected


def test_clip_q_symbolic_and_numeric_agree():
    """
    Given the same finite mass ratios,
    When both the symbolic and the numeric helper are applied,
    Then they agree exactly -- the Op path (numeric) and the graph path
      (symbolic) must not sanitize q differently.
    """
    for v in (1e-12, Q_MIN, 1e-4, 1.0, 50.0, Q_MAX, 1e4):
        assert float(clip_q(pt.as_tensor_variable(v)).eval()) == clip_q_value(
            v
        )


# ---------------------------------------------------------------------------
# The NaN path: propagate (symbolic) / raise by name (numeric), never invent
# ---------------------------------------------------------------------------


def test_clip_q_propagates_nan_instead_of_inventing_a_mass_ratio():
    """
    Given a NaN mass ratio,
    When clip_q is applied,
    Then the result is NaN, NOT Q_MIN.

    This is the substance of review item 4.5.  The old expression was
    ``clip(nan_to_num(q, nan=1e-9), 1e-9, 100)``, which reported q = 1e-9 --
    a fabricated mass ratio -- and, because nan_to_num is a switch, a zero
    gradient, so a failed computation produced a healthy-looking likelihood.
    q is NaN only when one of the masses it divides is already NaN, and that
    already makes the total logp NaN (test below), so nothing was ever
    rescued: the scrub only deleted the evidence.
    """
    assert np.isnan(float(clip_q(pt.as_tensor_variable(np.nan)).eval()))


def test_clip_q_value_raises_naming_the_parameter():
    """
    Given a NaN mass ratio on the numeric (Op / bootstrap) path,
    When clip_q_value is applied,
    Then it raises ValueError naming q and the parameters to check, rather
      than handing NaN to MulensModel -- which used to report the generic
      "Wrong number of solutions to the lens equation" three frames away.
    """
    with pytest.raises(ValueError) as exc:
        clip_q_value(np.nan, "lens.q")
    msg = str(exc.value)
    assert "lens.q" in msg
    assert "logmass" in msg and "log_q" in msg
    assert "must be a number" in msg


def test_binary_op_reports_a_nan_q_by_name_and_still_rejects_the_proposal():
    """
    Given a param vector whose q is NaN,
    When BinaryLensMagOp.perform runs,
    Then it still returns all-NaN magnifications (logp = -inf, proposal
      rejected -- byte-identical to the old behaviour), but the warn-once
      message now names lens.q instead of quoting a MulensModel internal.
    """
    p = np.array([2458554.89, 0.1, 18.2, 0.02, -0.01, 0.98, np.nan, -52.0])
    t = np.linspace(2458554.89 - 5, 2458554.89 + 5, 21)
    obs = np.zeros((len(t), 3))
    out = [[None]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BinaryLensMagOp(
            coords=_COORDS, mag_method="auto_vbbl", use_rho=False
        ).perform(None, [p, t, obs], out)

    assert np.all(np.isnan(out[0][0]))
    assert any("lens.q" in str(w.message) for w in caught)


def test_vbm_direct_still_short_circuits_a_non_finite_param_vector():
    """
    Given a param vector containing a NaN,
    When VBMDirectMagOp.perform runs,
    Then it returns all-NaN via its own explicit non-finite guard.  That
      guard was hoisted above the per-companion unpacking so it, not
      clip_q_value, is what notices -- previously it sat *after* a
      np.clip that silently passed NaN through.
    """
    p = np.array([2458554.89, 0.1, 18.2, 0.02, -0.01, 0.98, np.nan, -52.0])
    t = np.linspace(2458554.89 - 5, 2458554.89 + 5, 21)
    obs = np.zeros((len(t), 3))
    out = [[None]]
    VBMDirectMagOp(coords=_COORDS, n_companions=1, use_rho=False).perform(
        None, [p, t, obs], out
    )
    assert np.all(np.isnan(out[0][0]))


def test_build_binary_model_clips_a_finite_out_of_range_q():
    """
    Given a finite q above Q_MAX,
    When the MulensModel binary model is built,
    Then q is clamped to Q_MAX -- the range decision survives the removal of
      the NaN scrub.
    """
    p = np.array([2458554.89, 0.1, 18.2, 0.0, 0.0, 0.98, 1e6, -52.0])
    model = _build_binary_model(p, _COORDS, "auto_vbbl", use_rho=False)
    assert float(model.parameters.q) == Q_MAX


# ---------------------------------------------------------------------------
# Why the NaN branch is unreachable, pinned
# ---------------------------------------------------------------------------


def test_q_stays_finite_over_the_whole_sampled_support(binary_system):
    """
    Given a binary-lens model,
    When q is evaluated at raw coordinates far beyond anything a sampler
      reaches (+/-1e12 on every raw variable at once),
    Then it is always finite: both masses descend from logit-bounded
      coordinates and the denominator is a stellar mass with a hard 1e-9
      solMass floor.  This is the premise the nan_to_num removal rests on.
    """
    system, model = binary_system
    ip = model.initial_point()
    rvs = list(model.value_vars)
    q_node = model.replace_rvs_by_values([system.lens.q.value])[0]
    fn = pytensor.function(rvs, q_node, on_unused_input="ignore")

    base = [np.asarray(ip[v.name], dtype=float) for v in rvs]
    for extreme in (-1e12, -1e4, -50.0, 0.0, 50.0, 1e4, 1e12):
        vals = fn(*[np.full(np.shape(b), extreme) for b in base])
        assert np.all(np.isfinite(vals)), f"q not finite at raw={extreme}"

    rng = np.random.default_rng(0)
    for scale in (1.0, 1e3, 1e6):
        for _ in range(50):
            vals = fn(
                *[b + rng.normal(scale=scale, size=np.shape(b)) for b in base]
            )
            assert np.all(np.isfinite(vals))


def test_a_nan_raw_coordinate_already_makes_logp_nan(binary_system):
    """
    Given a NaN in a mass-carrying raw coordinate -- the only way q can go
      NaN,
    When the model logp is evaluated,
    Then it is NaN regardless of what q does, because that raw variable's own
      N(0, 1) prior term is NaN.  The proposal is rejected either way, so
      scrubbing q could never have rescued a sample; it could only hide which
      quantity failed.
    """
    _, model = binary_system
    ip = model.initial_point()
    logp = model.compile_logp()
    assert np.isfinite(logp(ip))

    mass_raws = [
        v.name
        for v in model.value_vars
        if "logmass" in v.name or "log_q" in v.name or ".mass" in v.name
    ]
    assert mass_raws, "expected a sampled mass coordinate in this topology"
    for name in mass_raws:
        point = dict(ip)
        point[name] = np.full(np.shape(np.asarray(ip[name])), np.nan)
        assert np.isnan(logp(point)), f"{name}=NaN did not poison logp"


# ---------------------------------------------------------------------------
# One implementation, used by both magnification paths
# ---------------------------------------------------------------------------


def test_binary_mm_params_q_is_the_q_parameter_clipped(binary_system):
    """
    Given the MulensModel-backend param builder,
    When it reports q,
    Then it is clip_q(lens.q), bit-identical to the mass ratio it used to
      recompute inline as ``m_companion / max(m_primary, 1e-10)``.  The
      pt.maximum floor was provably dead (star.mass = 10**logmass with
      logmass >= -9 dex is never below 1e-9), and going through the Parameter
      means the backend, the priors and the reports all see one q.
    """
    system, model = binary_system
    lens = system.lens

    built = lens._get_binary_mm_params(index=0)["q"]
    expected = clip_q(lens.q.value[0])

    m1 = system.star.mass.value[lens.lens_bodies[0][0][1]]
    l2_type, l2_idx = lens.lens_bodies[0][1]
    m2 = getattr(system, l2_type).mass.value[l2_idx]
    legacy = pt.clip(
        pt.nan_to_num(m2 / pt.maximum(m1, 1e-10), nan=1e-9), 1e-9, 100.0
    )

    rvs = list(model.value_vars)
    ip = model.initial_point()
    outs = model.replace_rvs_by_values([built, expected, legacy])
    fn = pytensor.function(rvs, outs, on_unused_input="ignore")
    got, want, old = fn(*[np.asarray(ip[v.name]) for v in rvs])

    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
    assert (
        np.asarray(got).ravel().tobytes() == np.asarray(old).ravel().tobytes()
    )


def test_alpha_deg_has_one_implementation(binary_system):
    """
    Given Lens._alpha_deg,
    When it is compared with the open-coded arctan2 the two call sites used,
    Then they are bit-identical -- lens.alpha's expression IS that arctan2
      (physics.calc_alpha), so reading the Parameter changes nothing except
      that there is now one spelling.
    """
    system, model = binary_system
    lens = system.lens

    consolidated = lens._alpha_deg(0)
    legacy = pt.arctan2(lens.yalpha.value[0], lens.xalpha.value[0]) * (
        180.0 / np.pi
    )
    from_params = lens._get_binary_mm_params(index=0)["alpha"]

    rvs = list(model.value_vars)
    ip = model.initial_point()
    outs = model.replace_rvs_by_values([consolidated, legacy, from_params])
    fn = pytensor.function(rvs, outs, on_unused_input="ignore")
    a, b, c = fn(*[np.asarray(ip[v.name]) for v in rvs])

    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert np.asarray(a).tobytes() == np.asarray(c).tobytes()


def test_alpha_parameter_expression_is_the_arctan2():
    """
    Given calc_alpha, the expression behind lens.alpha,
    When evaluated,
    Then it is arctan2(yalpha, xalpha) in radians -- the invariant
      _alpha_deg relies on to be a pure unit conversion.
    """
    x, y = pt.dscalar("x"), pt.dscalar("y")
    f = pytensor.function([x, y], calc_alpha(x, y))
    for xv, yv in ((0.6, -0.8), (-1.0, 0.0), (0.3, 0.4)):
        assert f(xv, yv) == np.arctan2(yv, xv)


def test_no_call_site_re_derives_q_or_alpha():
    """
    Given the two magnification param builders,
    When their source is inspected,
    Then neither scrubs q with nan_to_num nor re-derives alpha with arctan2:
      the duplication review item 4.5 named is gone, and cannot creep back by
      copy-paste without failing here.
    """
    for fn in (Lens._get_binary_mm_params, Lens.get_magnification_op):
        src = inspect.getsource(fn)
        assert "nan_to_num" not in src, (
            f"{fn.__name__} re-introduced the scrub"
        )
        assert "arctan2" not in src, f"{fn.__name__} re-derives alpha_deg"
        assert "1e-9" not in src, f"{fn.__name__} open-codes the q range"


def test_alpha_deg_reads_the_shared_rad_to_deg_constant():
    """
    Given Lens._alpha_deg,
    When its source is inspected,
    Then it converts through the module-level constant rather than spelling
      180/pi inline, which is how the two call sites drifted apart in the
      first place.
    """
    assert lens_mod._RAD_TO_DEG == 180.0 / np.pi
    src = inspect.getsource(Lens._alpha_deg)
    assert "_RAD_TO_DEG" in src
    assert "np.pi" not in src


# ---------------------------------------------------------------------------
# The build-time guard on the start value
# ---------------------------------------------------------------------------


def test_out_of_range_q_start_warns_and_says_what_to_do(binary_system, caplog):
    """
    Given a mass-ratio start above Q_MAX,
    When the lens builds its likelihood,
    Then it warns that the fit will actually START at the clipped value and
      names the parameter -- silently starting somewhere other than the seed
      is exactly what the runtime clip used to hide.
    """
    system, _ = binary_system
    lens = system.lens
    saved = lens.q.initval
    try:
        lens.q.initval = np.array([1e6])
        with caplog.at_level("WARNING"):
            lens._validate_q_start()
    finally:
        lens.q.initval = saved

    text = caplog.text
    assert "lens.q" in text or f"{lens.prefix}.q" in text
    assert "START" in text


def test_nan_q_start_raises(binary_system):
    """
    Given a NaN mass-ratio start (a broken seed, or a hard link whose
      expression went NaN),
    When the lens builds its likelihood,
    Then it raises, naming q and the masses to check.  This is the one place
      a raise is free: it is a check on the inputs at build time, not a
      mid-graph assert that would kill a run over a proposal the sampler
      already rejects on its own.
    """
    system, _ = binary_system
    lens = system.lens
    saved = lens.q.initval
    try:
        lens.q.initval = np.array([np.nan])
        with pytest.raises(ValueError) as exc:
            lens._validate_q_start()
    finally:
        lens.q.initval = saved

    msg = str(exc.value)
    assert ".q" in msg and "not a number" in msg
    assert "logmass" in msg


def test_infinite_q_start_warns_rather_than_raising(binary_system, caplog):
    """
    Given an infinite mass-ratio start,
    When the guard runs,
    Then it takes the out-of-range WARNING branch, not the raise -- the same
      NaN/infinity split clip_q_value makes, so the build-time guard and the
      runtime helper cannot disagree about what is fatal.
    """
    system, _ = binary_system
    lens = system.lens
    saved = lens.q.initval
    try:
        lens.q.initval = np.array([np.inf])
        with caplog.at_level("WARNING"):
            lens._validate_q_start()
    finally:
        lens.q.initval = saved
    assert ".q starts at" in caplog.text


def test_a_healthy_binary_start_neither_warns_nor_raises(
    binary_system, caplog
):
    """
    Given an ordinary binary-lens start,
    When the guard runs,
    Then it is silent -- the guard must not fire on working configs.
    """
    system, _ = binary_system
    with caplog.at_level("WARNING"):
        system.lens._validate_q_start()
    assert ".q starts at" not in caplog.text
