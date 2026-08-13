"""Trajectory-parameter (t_0, u_0, t_E, theta_E, pi_E_N, pi_E_E) sanitization.

The sequel to tests/test_q_sanitization.py.  ``Lens._get_safe_mm_params`` used
to scrub all five of its inputs with ``pt.nan_to_num``::

    t_E -> 100 d,  u_0 -> 1,  theta_E -> 0,  pi_E_N -> 0,  pi_E_E -> 0

so a fully-NaN parameter vector produced a complete, fabricated PSPL model and
a healthy-looking likelihood.  The scrub is gone; the three RANGE decisions it
was tangled up with (the t_E floor, the |u_0| floor, the no-lensing parallax
gate) survive, and the failure now names the parameter -- in the graph by
propagating to logp, on the numeric Op path by raising, and at build time by
``Lens._validate_pspl_start``.

The last section covers the sequel: the |u_0| floor was TWO different numbers
(1e-6 symbolic, a hard-coded 1e-9 on the Op path, plus a third copy in the flux
bootstrap), and in all three spellings it failed to engage at u_0 = 0 --
``sign(0) = 0`` -- which is the one value it exists to protect.  One constant,
one expression, 1e-9, and zero floors to +U_0_FLOOR.
"""

import functools
import inspect
import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.mulensing import lens as lens_module
from exozippy.components.mulensing import mulensinstrument as mi_module
from exozippy.components.mulensing import op as op_module
from exozippy.components.mulensing.lens import Lens
from exozippy.components.mulensing.mulensinstrument import MulensInstrument
from exozippy.components.mulensing.op import (
    MulensMagOp,
    VBMDirectMagOp,
    _base_mm_params,
    _build_pspl_model,
)
from exozippy.components.mulensing.physics import (
    T_E_FLOOR,
    THETA_E_LENSING_MIN,
    U_0_FLOOR,
    apply_u_0_floor,
    floor_u_0_value,
    require_mm_number,
)
from exozippy.system import System

_COORDS = "268.0d -29.0d"

# The floor the SYMBOLIC path used before the two were unified.  Kept as a
# named constant so the byte-identity test below can say exactly where the
# pre-#142 expression and the current one are allowed to differ.
_LEGACY_U_0_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pspl_config():
    """A minimal PSPL topology: star.0 lens, star.1 source."""
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [
            {
                "name": "Lens",
                "lenses": ["star.0"],
                "sources": ["star.1"],
                "finite_source": False,
            }
        ],
    }
    user_params = {
        "lens.Lens.t_0": {"initval": 2455379.571},
        "lens.Lens.u_0": {"initval": 0.523},
        "lens.Lens.t_E": {"initval": 17.94},
        "star.Lens.ra": {"initval": 268.0, "sigma": 0},
        "star.Lens.dec": {"initval": -29.0, "sigma": 0},
        "star.Source.ra": {"initval": 268.0, "sigma": 0},
        "star.Source.dec": {"initval": -29.0, "sigma": 0},
    }
    return config, user_params


@pytest.fixture(scope="module")
def pspl_system():
    config, user_params = _pspl_config()
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    return system, model


class _FakeParam:
    """The only thing _get_safe_mm_params asks of a Parameter: ``.value``."""

    def __init__(self, node):
        self.value = node


class _FakeLens:
    """Enough of a Lens to call _get_safe_mm_params on symbolic inputs."""

    def __init__(self):
        self.t_0 = _FakeParam(pt.dvector("t_0"))
        self.u_0 = _FakeParam(pt.dvector("u_0"))
        self.t_E = _FakeParam(pt.dvector("t_E"))
        self.theta_E = _FakeParam(pt.dvector("theta_E"))
        self.pi_E_N = _FakeParam(pt.dvector("pi_E_N"))
        self.pi_E_E = _FakeParam(pt.dvector("pi_E_E"))

    def inputs(self):
        return [
            self.t_0.value,
            self.u_0.value,
            self.t_E.value,
            self.theta_E.value,
            self.pi_E_N.value,
            self.pi_E_E.value,
        ]


_KEYS = ("t0", "u0", "tE", "pi_N", "pi_E")


def _legacy_safe_mm_params(lens, index=0):
    """The pre-fix body of Lens._get_safe_mm_params, verbatim."""
    tE_raw = lens.t_E.value[index]
    u0_raw = lens.u_0.value[index]
    theta_E_raw = lens.theta_E.value[index]
    pi_N_raw = lens.pi_E_N.value[index]
    pi_E_raw = lens.pi_E_E.value[index]

    tE_scrubbed = pt.nan_to_num(tE_raw, nan=100.0)
    u0_scrubbed = pt.nan_to_num(u0_raw, nan=1.0)
    theta_E_scrubbed = pt.nan_to_num(theta_E_raw, nan=0.0)
    pi_N_scrubbed = pt.nan_to_num(pi_N_raw, nan=0.0)
    pi_E_scrubbed = pt.nan_to_num(pi_E_raw, nan=0.0)

    tE_safe = pt.maximum(tE_scrubbed, 1e-4)
    u0_safe = pt.sign(u0_scrubbed) * pt.maximum(pt.abs(u0_scrubbed), 1e-6)
    is_physical = pt.gt(theta_E_scrubbed, 1e-6)

    return {
        "t0": lens.t_0.value[index],
        "u0": u0_safe,
        "tE": tE_safe,
        "pi_N": pt.switch(is_physical, pi_N_scrubbed, 0.0),
        "pi_E": pt.switch(is_physical, pi_E_scrubbed, 0.0),
    }


def _compiled_pair():
    """(new, legacy) evaluators of the five outputs on the same inputs."""
    fake = _FakeLens()
    new = Lens._get_safe_mm_params(fake, 0)
    old = _legacy_safe_mm_params(fake, 0)
    ins = fake.inputs()
    f_new = pytensor.function(ins, [new[k] for k in _KEYS])
    f_old = pytensor.function(ins, [old[k] for k in _KEYS])
    return f_new, f_old


# ---------------------------------------------------------------------------
# The surviving range decisions are unchanged, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vals",
    [
        (2455379.571, 0.523, 17.94, 0.2337, 0.0767, 0.0485),
        (2456836.22, -0.922, 25.03, 0.878, -0.2706, 0.2074),
        (2458554.89, 1e-5, 1e-3, 1e-5, 0.0, 0.0),
        (2458554.89, -1e-7, 1e-6, 1e-9, 1.0, -1.0),
        (2458554.89, 18.0, 1e19, 5e4, 1e18, -1e18),
        (2458554.89, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2458554.89, 0.5, -30.0, -1.0, 3.0, 4.0),
    ],
)
def test_finite_inputs_are_byte_identical_to_the_old_expression(vals):
    """
    Given any FINITE trajectory parameters -- in range, out of range, at the
      floors, and at the extremes of the sampled support,
    When the five sanitized outputs are computed,
    Then they are bit-identical to the pre-fix expression that scrubbed with
      nan_to_num first.  Removing the scrub is a change to the NaN case only;
      no working fit may move.

    The ONE documented exception is ``u0`` inside the legacy 1e-6 floor: the
    two floors have since been unified at 1e-9 (the Op path's), and exactly
    zero now floors to +U_0_FLOOR instead of being missed by ``sign(0) = 0``.
    Outside that band -- which is everything a real event visits -- u0 too is
    bit-identical, and it is asserted as such here rather than exempted.
    """
    f_new, f_old = _compiled_pair()
    args = [np.array([v], dtype=float) for v in vals]
    new, old = f_new(*args), f_old(*args)
    u_0 = vals[1]

    for k, a, b in zip(_KEYS, new, old):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if k == "u0" and abs(u_0) < _LEGACY_U_0_FLOOR:
            # The floors used to disagree here; pin the new value outright.
            assert float(a) == floor_u_0_value(u_0)
            continue
        assert a.tobytes() == b.tobytes(), f"{k}: {a} != {b}"


def test_the_floors_are_the_documented_constants():
    """
    Given the three range decisions, named constants in physics.py,
    When their values are read,
    Then t_E's and theta_E's are the literals that were open-coded in the old
      expression -- naming a constant must not move a number -- and |u_0|'s is
      1e-9, the value the two paths were UNIFIED at.

    U_0_FLOOR is the one that moved, deliberately: it was 1e-6 on the symbolic
    path against a hard-coded 1e-9 in ``op._base_mm_params``, so a fit visiting
    ``1e-9 <= |u_0| < 1e-6`` got a different answer depending on which backend
    it was on.  The looser of the two wins: the floor is a validity limit and
    the model should be clamped as little as float64 allows.
    """
    assert T_E_FLOOR == 1e-4
    assert THETA_E_LENSING_MIN == 1e-6
    assert U_0_FLOOR == 1e-9


def test_t_E_is_floored_and_u_0_keeps_its_sign():
    """
    Given a non-positive t_E and a tiny negative u_0,
    When the range decisions are applied,
    Then t_E is floored at T_E_FLOOR and |u_0| at U_0_FLOOR with the sign
      preserved -- these are statements about where the backends are defined
      and they survive the removal of the NaN scrub.
    """
    f_new, _ = _compiled_pair()
    out = dict(
        zip(
            _KEYS,
            f_new(
                np.array([2458554.89]),
                np.array([-1e-12]),
                np.array([-5.0]),
                np.array([1.0]),
                np.array([0.1]),
                np.array([0.2]),
            ),
        )
    )
    assert float(out["tE"]) == T_E_FLOOR
    assert float(out["u0"]) == -U_0_FLOOR


def test_the_parallax_gate_still_zeroes_pi_E_below_the_lensing_threshold():
    """
    Given a theta_E at or below THETA_E_LENSING_MIN (no lensing) with a
      nonzero parallax vector,
    When the params are built,
    Then pi_E_N and pi_E_E are zero -- pi_E = pi_rel/theta_E diverges there,
      and the soft bounds in build_likelihood, not a fabricated value, are
      what push the sampler out.
    """
    f_new, _ = _compiled_pair()
    out = dict(
        zip(
            _KEYS,
            f_new(
                np.array([2458554.89]),
                np.array([0.5]),
                np.array([20.0]),
                np.array([THETA_E_LENSING_MIN]),
                np.array([0.3]),
                np.array([-0.4]),
            ),
        )
    )
    assert float(out["pi_N"]) == 0.0
    assert float(out["pi_E"]) == 0.0


# ---------------------------------------------------------------------------
# The NaN path: propagate, never fabricate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "which, fabricated, affected",
    [
        ("t_E", 100.0, ["tE"]),
        ("u_0", 1.0, ["u0"]),
        ("pi_E_N", 0.0, ["pi_N"]),
        ("pi_E_E", 0.0, ["pi_E"]),
    ],
)
def test_a_nan_input_propagates_instead_of_becoming_its_old_default(
    which, fabricated, affected
):
    """
    Given a NaN in one trajectory parameter,
    When the params are built,
    Then the corresponding output is NaN, NOT the value the old nan_to_num
      substituted (t_E -> 100 d, u_0 -> 1, pi_E -> 0).

    This is the substance of the change.  A NaN here means an input is already
    NaN, i.e. the raw vector carries a NaN, which already makes the total logp
    NaN through that variable's own N(0, 1) prior (test below) -- so the
    proposal is rejected either way and the substitution could never rescue a
    sample.  What it did instead was invent an entire event geometry, with a
    zero gradient (nan_to_num is a switch), in place of the one quantity that
    would have named the failure.
    """
    f_new, f_old = _compiled_pair()
    base = dict(
        t_0=2458554.89,
        u_0=0.523,
        t_E=17.94,
        theta_E=0.234,
        pi_E_N=0.077,
        pi_E_E=0.049,
    )
    base[which] = np.nan
    args = [
        np.array([base[k]])
        for k in ("t_0", "u_0", "t_E", "theta_E", "pi_E_N", "pi_E_E")
    ]

    new = dict(zip(_KEYS, f_new(*args)))
    old = dict(zip(_KEYS, f_old(*args)))
    for key in affected:
        assert np.isnan(float(new[key])), f"{key} did not propagate the NaN"
        assert float(old[key]) == fabricated, "legacy expression changed"


def test_a_nan_theta_E_still_gates_the_parallax_off_and_nans_t_E():
    """
    Given a NaN theta_E,
    When the params are built,
    Then the parallax gate is False (a comparison against NaN is False, with
      or without the old nan_to_num -- which is why THAT substitution was a
      no-op in every case, NaN included) and the NaN still reaches the model
      through t_E, which is theta_E / mu_rel.
    """
    f_new, f_old = _compiled_pair()
    args = [
        np.array([2458554.89]),
        np.array([0.523]),
        np.array([np.nan]),  # t_E, which derives from theta_E
        np.array([np.nan]),  # theta_E
        np.array([0.077]),
        np.array([0.049]),
    ]
    new = dict(zip(_KEYS, f_new(*args)))
    old = dict(zip(_KEYS, f_old(*args)))
    assert float(new["pi_N"]) == 0.0 and float(new["pi_E"]) == 0.0
    assert float(old["pi_N"]) == 0.0 and float(old["pi_E"]) == 0.0
    assert np.isnan(float(new["tE"]))
    assert float(old["tE"]) == 100.0


def test_no_scrub_survives_in_the_source():
    """
    Given _get_safe_mm_params,
    When its source is inspected,
    Then it neither calls nan_to_num nor open-codes the three floors, so the
      fabrication cannot creep back by copy-paste without failing here.
    """
    src = inspect.getsource(Lens._get_safe_mm_params)
    body = src.split('"""')[-1]
    assert "nan_to_num" not in body
    assert "1e-4" not in body
    assert "1e-6" not in body and "1e-9" not in body


# ---------------------------------------------------------------------------
# Why the NaN branch is unreachable, pinned on a real model
# ---------------------------------------------------------------------------


def test_the_five_stay_finite_over_the_whole_sampled_support(pspl_system):
    """
    Given a PSPL model,
    When the five quantities are evaluated at raw coordinates far beyond
      anything a sampler reaches (+/-1e12 on every raw variable at once, then
      one at a time, then randomly),
    Then they are always finite.  t_0/u_0 are logit-bounded; theta_E's
      radicand is floored, so it is strictly positive; and t_E and pi_E are
      ratios whose denominators are floored at THETA_E_FLOOR / MU_REL_FLOOR.
      This is the premise the removal rests on.
    """
    system, model = pspl_system
    lens = system.lens
    names = ["t_0", "u_0", "t_E", "theta_E", "pi_E_N", "pi_E_E"]
    rvs = list(model.value_vars)
    outs = model.replace_rvs_by_values([getattr(lens, n).value for n in names])
    fn = pytensor.function(rvs, outs, on_unused_input="ignore")

    ip = model.initial_point()
    base = [np.asarray(ip[v.name], dtype=float) for v in rvs]

    with np.errstate(all="ignore"):
        for extreme in (-1e12, -1e4, -50.0, 0.0, 50.0, 1e4, 1e12):
            vals = fn(*[np.full(np.shape(b), extreme) for b in base])
            for n, v in zip(names, vals):
                assert np.all(np.isfinite(v)), f"{n} not finite at {extreme}"

        for i in range(len(rvs)):
            for extreme in (-1e12, 1e12):
                args = [b.copy() for b in base]
                args[i] = np.full(np.shape(base[i]), extreme)
                for n, v in zip(names, fn(*args)):
                    assert np.all(np.isfinite(v)), (
                        f"{n} not finite at {rvs[i].name}={extreme}"
                    )

        rng = np.random.default_rng(0)
        for scale in (1.0, 1e3, 1e6):
            for _ in range(40):
                vals = fn(
                    *[
                        b + rng.normal(scale=scale, size=np.shape(b))
                        for b in base
                    ]
                )
                for n, v in zip(names, vals):
                    assert np.all(np.isfinite(v)), f"{n} not finite (random)"


def test_a_nan_raw_coordinate_already_makes_logp_nan(pspl_system):
    """
    Given a NaN in any sampled coordinate -- the only way these five can go
      NaN,
    When the model logp is evaluated,
    Then it is NaN regardless of what the trajectory params do, because that
      raw variable's own N(0, 1) prior term is NaN.  The proposal is rejected
      either way, so the scrub could never have rescued a sample; it could
      only hide which quantity failed.
    """
    _, model = pspl_system
    ip = model.initial_point()
    logp = model.compile_logp()
    assert np.isfinite(logp(ip))

    for v in model.value_vars:
        point = dict(ip)
        point[v.name] = np.full(np.shape(np.asarray(ip[v.name])), np.nan)
        assert np.isnan(logp(point)), f"{v.name}=NaN did not poison logp"


# ---------------------------------------------------------------------------
# The numeric (Op) path names the parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index, label",
    [
        (0, "lens.t_0"),
        (1, "lens.u_0"),
        (2, "lens.t_E"),
        (3, "lens.pi_E_N"),
        (4, "lens.pi_E_E"),
    ],
)
def test_base_mm_params_raises_naming_the_nan_parameter(index, label):
    """
    Given a param vector with a NaN in one trajectory slot,
    When the MulensModel param builder runs,
    Then it raises ValueError naming that parameter, instead of handing NaN
      to MulensModel and letting it report whatever generic failure the NaN
      happened to cause several frames away.
    """
    p = np.array([2458554.89, 0.1, 18.2, 0.02, -0.01])
    p[index] = np.nan
    with pytest.raises(ValueError) as exc:
        _base_mm_params(p)
    msg = str(exc.value)
    assert label in msg
    assert "must be a number" in msg
    assert "logmass" in msg and "distance" in msg


def test_base_mm_params_is_unchanged_for_finite_input():
    """
    Given finite trajectory parameters,
    When _base_mm_params runs,
    Then every field is what the old body produced: t_0 and the two pi_E
      untouched, |u_0| floored at U_0_FLOOR with its sign, t_E floored at
      T_E_FLOOR.
    """
    p = np.array([2458554.89, -1e-12, -3.0, 0.02, -0.01])
    got = _base_mm_params(p)
    assert got["t_0"] == 2458554.89
    assert got["u_0"] == -U_0_FLOOR
    assert got["t_E"] == T_E_FLOOR
    assert got["pi_E_N"] == 0.02
    assert got["pi_E_E"] == -0.01


def test_require_mm_number_passes_infinities_through():
    """
    Given an infinite trajectory parameter,
    When require_mm_number is applied,
    Then it is returned, not raised on -- the same NaN/infinity split
      clip_q_value makes.  An infinity carries a sign and is an ordinary
      out-of-range value the caller's own floor handles; NaN has no value at
      all.
    """
    assert require_mm_number(np.inf, "lens.t_E") == np.inf
    assert require_mm_number(-np.inf, "lens.u_0") == -np.inf
    assert (
        _base_mm_params(np.array([2458554.89, 0.1, np.inf, 0.0, 0.0]))["t_E"]
        == np.inf
    )


def test_pspl_op_reports_a_nan_by_name_and_still_rejects_the_proposal():
    """
    Given a param vector whose t_E is NaN,
    When MulensMagOp.perform runs,
    Then it still returns all-NaN magnifications (logp = -inf, proposal
      rejected -- unchanged), but the warn-once message now names lens.t_E.
    """
    p = np.array([2458554.89, 0.1, np.nan, 0.0, 0.0])
    t = np.linspace(2458554.89 - 5, 2458554.89 + 5, 21)
    obs = np.zeros((len(t), 3))
    out = [[None]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MulensMagOp(coords=_COORDS, mag_method="point_source").perform(
            None, [p, t, obs], out
        )

    assert np.all(np.isnan(out[0][0]))
    assert any("lens.t_E" in str(w.message) for w in caught)


def test_build_pspl_model_is_unchanged_for_a_healthy_vector():
    """
    Given an ordinary PSPL param vector,
    When the MulensModel model is built,
    Then its parameters are exactly the input -- the guard is inert on
      everything a fit actually visits.
    """
    p = np.array([2458554.89, 0.1, 18.2, 0.02, -0.01])
    model = _build_pspl_model(p, _COORDS, "point_source", use_rho=False)
    assert float(model.parameters.t_0) == 2458554.89
    assert float(model.parameters.u_0) == 0.1
    assert float(model.parameters.t_E) == 18.2
    assert float(model.parameters.pi_E_N) == 0.02
    assert float(model.parameters.pi_E_E) == -0.01


def test_vbm_direct_names_the_non_finite_entry_it_short_circuits_on():
    """
    Given a param vector with a NaN t_E,
    When VBMDirectMagOp.perform runs,
    Then it returns all-NaN through its own non-finite guard -- unchanged --
      but warns once naming lens.t_E.  That branch used to return NaN in
      complete silence, which is indistinguishable from an ordinary rejected
      proposal even when the model is misconfigured on every one.
    """
    p = np.array([2458554.89, 0.1, np.nan, 0.02, -0.01, 0.98, 1.1e-3, -52.0])
    t = np.linspace(2458554.89 - 5, 2458554.89 + 5, 21)
    obs = np.zeros((len(t), 3))
    out = [[None]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        VBMDirectMagOp(coords=_COORDS, n_companions=1, use_rho=False).perform(
            None, [p, t, obs], out
        )

    assert np.all(np.isnan(out[0][0]))
    assert any("lens.t_E" in str(w.message) for w in caught)


def test_vbm_direct_labels_cover_the_whole_param_vector():
    """
    Given the VBM param layouts,
    When the label list is built,
    Then it lines up with the vector the Op actually unpacks, so the warning
      cannot name the wrong parameter.
    """
    op = VBMDirectMagOp(coords=_COORDS, n_companions=2, use_rho=True)
    labels = op._param_labels()
    assert labels[:5] == [
        "lens.t_0",
        "lens.u_0",
        "lens.t_E",
        "lens.pi_E_N",
        "lens.pi_E_E",
    ]
    assert labels[5] == "lens.rho"
    assert labels[6:9] == ["lens.s[0]", "lens.q[0]", "lens.alpha[0]"]
    assert labels[9:12] == ["lens.s[1]", "lens.q[1]", "lens.alpha[1]"]
    assert len(labels) == 12


# ---------------------------------------------------------------------------
# The build-time guard on the start values
# ---------------------------------------------------------------------------


def test_a_healthy_pspl_start_neither_warns_nor_raises(pspl_system, caplog):
    """
    Given an ordinary PSPL start,
    When the guard runs,
    Then it is silent -- it must not fire on working configs.
    """
    system, _ = pspl_system
    with caplog.at_level("WARNING"):
        system.lens._validate_pspl_start()
    assert "starts at" not in caplog.text


@pytest.mark.parametrize("name", ["t_0", "u_0"])
def test_a_nan_sampled_start_raises_naming_the_parameter(pspl_system, name):
    """
    Given a NaN start for a SAMPLED trajectory parameter (a broken seed, or a
      hard link whose expression went NaN),
    When the lens builds its likelihood,
    Then it raises, naming that parameter and what to check.  This is the one
      place a raise is free: a check on the inputs at build time, not a
      mid-graph assert that would kill a run over a proposal the sampler
      already rejects on its own.
    """
    system, _ = pspl_system
    par = getattr(system.lens, name)
    saved = par.initval
    try:
        par.initval = np.array([np.nan])
        with pytest.raises(ValueError) as exc:
            system.lens._validate_pspl_start()
    finally:
        par.initval = saved

    msg = str(exc.value)
    assert f".{name}" in msg
    assert "not a number" in msg
    assert "logmass" in msg


@pytest.mark.parametrize("name", ["t_E", "theta_E", "pi_E_N", "pi_E_E"])
def test_a_derived_parameters_initval_is_not_treated_as_the_start(
    pspl_system, name, caplog
):
    """
    Given a NaN in the resolved ``initval`` of a DERIVED trajectory parameter,
    When the guard runs,
    Then it neither raises nor warns, because for a derived parameter
      ``initval`` is the relaxation engine's bookkeeping and NOT the value the
      model starts at -- the graph recomputes it from the sampled coordinates.

    This is not hypothetical.  examples/ob161003 ships with
    ``lens.theta_E.initval = [nan, 0.8393]`` (the engine only ever solved the
    second source slot) while the model starts at ``theta_E = [0.8393,
    0.8393]`` with a finite logp, so a guard that read derived initvals would
    refuse to build a working, shipped example.
    """
    system, _ = pspl_system
    par = getattr(system.lens, name)
    saved = par.initval
    try:
        par.initval = np.array([np.nan])
        with caplog.at_level("WARNING"):
            system.lens._validate_pspl_start()  # must not raise
    finally:
        par.initval = saved
    assert "starts at" not in caplog.text


@pytest.mark.parametrize("u0", [0.0, 1e-12, -1e-10])
def test_a_tiny_u_0_start_warns(pspl_system, caplog, u0):
    """
    Given an impact parameter seeded inside the |u_0| floor -- including
      exactly zero, a plausible seed for a high-magnification event,
    When the guard runs,
    Then it warns, and the message names the value the fit will actually START
      at.  Exactly zero now floors (to +U_0_FLOOR) rather than slipping through
      ``sign(0) = 0``, so the warning reports a seed/start mismatch instead of
      confessing that the floor gave up.
    """
    system, _ = pspl_system
    par = system.lens.u_0
    saved = par.initval
    try:
        par.initval = np.array([u0])
        with caplog.at_level("WARNING"):
            system.lens._validate_pspl_start()
    finally:
        par.initval = saved
    assert ".u_0 starts at" in caplog.text
    assert repr(floor_u_0_value(u0)) in caplog.text


def test_a_u_0_start_exactly_at_the_floor_does_not_warn(pspl_system, caplog):
    """
    Given a seed exactly at -U_0_FLOOR,
    When the guard runs,
    Then it is silent: the clip is inert there, so the fit starts exactly
      where the seed says and there is nothing to report.  The warning fires
      on a seed/start MISMATCH, not on smallness.
    """
    system, _ = pspl_system
    par = system.lens.u_0
    saved = par.initval
    try:
        par.initval = np.array([-U_0_FLOOR])
        with caplog.at_level("WARNING"):
            system.lens._validate_pspl_start()
    finally:
        par.initval = saved
    assert ".u_0 starts at" not in caplog.text


def test_an_infinite_start_does_not_raise(pspl_system, caplog):
    """
    Given an infinite u_0 start,
    When the guard runs,
    Then it does not raise -- the same NaN/infinity split require_mm_number
      and clip_q_value make, so the build-time guard and the runtime helpers
      cannot disagree about what is fatal.  An infinity carries a sign; NaN
      has no value at all.
    """
    system, _ = pspl_system
    par = system.lens.u_0
    saved = par.initval
    try:
        par.initval = np.array([-np.inf])
        with caplog.at_level("WARNING"):
            system.lens._validate_pspl_start()  # must not raise
    finally:
        par.initval = saved
    assert ".u_0 starts at" not in caplog.text


def test_the_examples_that_ship_build_without_the_guard_firing(pspl_system):
    """
    Given the guard,
    When it is asked which parameters it reads,
    Then it reads only the two SAMPLED trajectory parameters.  A guard that
      read the derived ones would refuse examples/ob161003, whose engine-
      resolved theta_E initval carries a NaN in the slot it never needed to
      solve while the model itself starts fine.
    """
    src = inspect.getsource(Lens._validate_pspl_start)
    body = src.split('"""')[-1]
    assert '"t_0"' in body and '"u_0"' in body
    for derived in ("t_E", "theta_E", "pi_E_N", "pi_E_E"):
        assert f'"{derived}"' not in body


# ---------------------------------------------------------------------------
# One floor, one expression, every path
#
# |u_0| used to be floored at 1e-6 on the symbolic path (physics.U_0_FLOOR)
# and at a hard-coded 1e-9 in op._base_mm_params -- three orders of magnitude
# apart, so a fit visiting 1e-9 <= |u_0| < 1e-6 got a different answer
# depending on which magnification backend it happened to be on.  A third copy
# of the clip (also 1e-9) sat in the flux bootstrap.  All three now call
# physics.apply_u_0_floor / physics.floor_u_0_value, unified at 1e-9.
#
# And all three shared a second defect: written as
# ``sign(u_0) * max(|u_0|, FLOOR)``, the floor did not engage at u_0 = 0 --
# ``sign(0) = 0`` gives ``0 * FLOOR = 0`` -- so the one value the floor exists
# to protect was the one it missed.  Zero now floors to +U_0_FLOOR.
# ---------------------------------------------------------------------------


def _u_0_grid():
    """|u_0| from 1e-12 to 1e-3, both signs, plus the boundary cases."""
    mags = [m * 10.0**e for e in range(-12, -2) for m in (1.0, 2.5, 5.0, 9.99)]
    vals = set(mags) | {-m for m in mags}
    vals |= {0.0, -0.0, 1e-3, -1e-3, U_0_FLOOR, -U_0_FLOOR}
    return np.array(sorted(vals))


class _FakeStarParam:
    def __init__(self, value):
        self.value = pt.as_tensor_variable(np.array([value], dtype=float))


class _FakeSystem:
    """Enough of a System for Lens.get_magnification: star.ra / star.dec."""

    def __init__(self, ra_rad, dec_rad):
        self.star = type(
            "S",
            (),
            {"ra": _FakeStarParam(ra_rad), "dec": _FakeStarParam(dec_rad)},
        )()


@functools.lru_cache(maxsize=1)
def _symbolic_A():
    """Compile the PRODUCTION symbolic magnification as a function of u_0.

    Uses ``Lens.get_magnification`` itself -- not a transcription of the
    Paczynski formula -- so the comparison below really is between the two
    backends and not between two copies of the same algebra.
    """
    fake = _FakeLens()
    fake.source_map = np.array([0])
    # get_magnification calls self._get_safe_mm_params -- bind the production
    # method so the clip under test is the one the model really uses.
    fake._get_safe_mm_params = functools.partial(
        Lens._get_safe_mm_params, fake
    )
    system = _FakeSystem(np.deg2rad(268.0), np.deg2rad(-29.0))
    times = pt.dvector("times")
    obs = pt.dmatrix("obs")
    node = Lens.get_magnification(fake, times, obs, system, 0)
    fn = pytensor.function(
        fake.inputs() + [times, obs], node, on_unused_input="ignore"
    )

    def A(u_0, t_0=2458554.0, t_E=20.0, times_np=None):
        t = np.array([t_0]) if times_np is None else np.asarray(times_np)
        one = lambda v: np.array([v], dtype=float)  # noqa: E731
        return np.asarray(
            fn(
                one(t_0),
                one(u_0),
                one(t_E),
                one(1.0),  # theta_E: above THETA_E_LENSING_MIN
                one(0.0),
                one(0.0),
                t,
                np.zeros((len(t), 3)),
            ),
            dtype=float,
        )

    return A


def _op_A(u_0, t_0=2458554.0, t_E=20.0, times_np=None):
    """The MulensModel backend's magnification, through _base_mm_params."""
    t = np.array([t_0]) if times_np is None else np.asarray(times_np)
    model = _build_pspl_model(
        np.array([t_0, u_0, t_E, 0.0, 0.0]),
        _COORDS,
        "point_source",
        use_rho=False,
    )
    return np.asarray(model.get_magnification(t), dtype=float)


def test_the_two_paths_apply_the_same_u_0_clip_bit_for_bit():
    """
    Given |u_0| across twelve decades, both signs, and both IEEE zeros,
    When the symbolic clip and the numeric one are applied,
    Then they return bit-identical values.  This is the unification itself:
      one constant, one expression, no second copy to drift.
    """
    grid = _u_0_grid()
    x = pt.dvector("x")
    sym = np.asarray(pytensor.function([x], apply_u_0_floor(x))(grid))
    num = np.array([floor_u_0_value(v) for v in grid])
    assert sym.tobytes() == num.tobytes()
    assert np.all(np.abs(sym) >= U_0_FLOOR)


def test_the_two_backends_agree_on_the_magnification_across_the_grid():
    """
    Given |u_0| from 1e-12 to 1e-3 -- spanning the whole band where the two
      floors used to disagree (1e-9 to 1e-6) -- and both signs,
    When the magnification is evaluated at peak on the symbolic path and on
      the MulensModel Op path,
    Then the two agree exactly.  Before the floors were unified the Op path
      saw the true u_0 and the symbolic path saw 1e-6 anywhere below 1e-6, so
      the same event had two peak magnifications up to 1000x apart depending
      on whether it was PSPL (symbolic, NUTS) or binary/finite-source (Op).
    """
    A_sym = _symbolic_A()
    for u_0 in _u_0_grid():
        a = float(A_sym(u_0)[0])
        b = float(_op_A(u_0)[0])
        assert np.isfinite(a) and np.isfinite(b), u_0
        assert abs(a - b) <= 1e-12 * abs(a), (
            f"u_0={u_0!r}: symbolic {a!r} vs Op {b!r}"
        )


def test_an_exactly_zero_u_0_gives_a_finite_magnification_on_both_paths():
    """
    Given u_0 = 0 -- an exactly central trajectory, and a perfectly plausible
      seed for a high-magnification event,
    When the magnification is evaluated AT t_0 on both paths,
    Then it is finite and equal to 1/U_0_FLOOR on both.

    This is the second bug the unification fixes.  ``sign(0) = 0`` made
    ``sign(u_0) * max(|u_0|, FLOOR)`` return 0, so the floor missed the one
    point it exists to protect and A(0) = inf reached the likelihood -- on the
    symbolic path, the Op path and the flux bootstrap alike.
    """
    A_sym = _symbolic_A()
    for zero in (0.0, -0.0):
        a = float(A_sym(zero)[0])
        b = float(_op_A(zero)[0])
        assert np.isfinite(a) and np.isfinite(b)
        assert a == pytest.approx(1.0 / U_0_FLOOR, rel=1e-12)
        assert b == pytest.approx(1.0 / U_0_FLOOR, rel=1e-12)


def test_zero_floors_positive_and_the_two_ieee_zeros_land_together():
    """
    Given +0.0 and -0.0,
    When the clip is applied,
    Then both give +U_0_FLOOR.

    The sign of a central crossing is genuinely undefined -- there is no side
    to be on -- and the two branches are the same event under the exact
    reflection (u_0, pi_E_N, pi_E_E) -> (-u_0, -pi_E_N, -pi_E_E), so either is
    defensible.  Positive is chosen because it keeps the map monotonically
    non-decreasing (0 is equidistant from the interval's two endpoints and
    ties break upward) and because ``-0.0 < 0`` is False, so the two IEEE
    zeros cannot land on opposite endpoints.
    """
    assert floor_u_0_value(0.0) == U_0_FLOOR
    assert floor_u_0_value(-0.0) == U_0_FLOOR
    x = pt.dvector("x")
    got = pytensor.function([x], apply_u_0_floor(x))(np.array([0.0, -0.0]))
    assert list(np.asarray(got)) == [U_0_FLOOR, U_0_FLOOR]


@pytest.mark.parametrize(
    "u_0", [U_0_FLOOR, 1e-8, 1e-6, 1e-3, 0.029, 0.14, 0.523, 1.7]
)
def test_the_reflection_symmetry_is_still_exact_on_both_paths(u_0):
    """
    Given a trajectory and its mirror image u_0 -> -u_0 (no parallax, so the
      two are the same event),
    When the magnification is evaluated on both paths,
    Then A(u_0) and A(-u_0) are bit-identical.

    The clip restores the sign for exactly this reason: the reflection is a
    real degeneracy (ob140939 has four Yee+2015 basins that differ by a sign
    flip), and a clip that broke it would make one branch of a genuine
    degeneracy unreachable.
    """
    A_sym = _symbolic_A()
    t = np.linspace(2458554.0 - 40.0, 2458554.0 + 40.0, 101)
    assert (
        A_sym(u_0, times_np=t).tobytes() == A_sym(-u_0, times_np=t).tobytes()
    )
    assert (
        _op_A(u_0, times_np=t).tobytes() == _op_A(-u_0, times_np=t).tobytes()
    )


def test_the_clip_matches_the_old_sign_abs_spelling_everywhere_but_zero():
    """
    Given the same floor in both spellings,
    When they are compared over the grid,
    Then they are bit-identical except at u_0 = 0, where the old one returned
      0 and the new one returns +U_0_FLOOR.

    Changing HOW the clip is written must not change WHAT it does; the whole
    behavioural difference has to be the one value that was broken.
    """
    grid = _u_0_grid()
    x = pt.dvector("x")
    new = np.asarray(pytensor.function([x], apply_u_0_floor(x))(grid))
    old = np.asarray(
        pytensor.function([x], pt.sign(x) * pt.maximum(pt.abs(x), U_0_FLOOR))(
            grid
        )
    )
    differ = [g for g, a, b in zip(grid, new, old) if a != b]
    assert differ == [0.0], differ
    assert float(old[list(grid).index(0.0)]) == 0.0


def test_the_clip_has_no_nan_gradient_and_no_nan_in_the_unselected_branch():
    """
    Given the switch-based clip,
    When its value and its gradient are evaluated at the boundaries, at both
      infinities and inside the floor,
    Then everything is finite -- a NaN in the UNSELECTED branch of a where is
      the JAX trap that freezes numpyro chains (see CLAUDE.md), and both
      branches here are min/max against a finite constant.  NaN INPUT still
      propagates, which is PR #142's rule: a floor is a range decision and
      must never double as a NaN substitution.
    """
    x = pt.dvector("x")
    y = apply_u_0_floor(x)
    val = pytensor.function([x], y)
    grad = pytensor.function([x], pt.grad(pt.sum(y), x))

    probe = np.array(
        [-1.0, -1e-3, -U_0_FLOOR, -1e-12, 0.0, 1e-12, U_0_FLOOR, 1e-3, 1.0]
    )
    assert np.all(np.isfinite(val(probe)))
    assert np.all(np.isfinite(grad(probe)))

    edge = np.array([np.inf, -np.inf])
    assert list(np.asarray(val(edge))) == [np.inf, -np.inf]
    assert np.all(np.isfinite(grad(edge)))

    assert np.isnan(float(np.asarray(val(np.array([np.nan])))[0]))


def test_the_floor_costs_no_precision_at_the_magnification_it_allows():
    """
    Given the loosest floor of the two, 1e-9,
    When the magnification there and the flux model built on it are computed,
    Then A(U_0_FLOOR) equals its analytic limit 1/U_0_FLOOR to the bit, and
      nothing downstream overflows.

    A -> 1/u as u -> 0, so A(1e-9) = 1e9 -- large, but the only term lost is
    the u^2 in (u^2 + 2), whose relative weight is 5e-19, three orders below
    eps = 2.2e-16.  Every operation in (u^2+2)/(u*sqrt(u^2+4)) is a product or
    a sum of positives, so there is no cancellation to be catastrophic, and
    the full float64 mantissa survives.  Loosening the floor from 1e-6 to 1e-9
    therefore buys three decades of unclamped model for no arithmetic at all.
    """
    u = U_0_FLOOR
    A = (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))
    assert A == 1.0 / u
    assert A == pytest.approx(1e9, rel=1e-12)

    # ... and the flux model / Gaussian logp built on it, across the three
    # flux zeropoints mulensinstrument actually sees (magnitude files ~1e-8,
    # difference imaging O(1) to O(1e4)).
    for f_s in (1e-8, 1.0, 1e4):
        F = f_s * A + f_s
        chi2 = ((F - f_s) / (0.01 * f_s)) ** 2
        assert np.isfinite(F) and np.isfinite(chi2)
        assert F < np.sqrt(np.finfo(float).max)


def test_the_flux_bootstrap_uses_the_shared_floor_too():
    """
    Given a central seed (u_0 = 0) and a time grid that contains t_0 exactly,
    When the flux bootstrap's PSPL magnification column is built,
    Then it is finite.

    This was the third hard-coded copy of the clip.  Unfloored, ``u_traj = 0``
    at ``t = t_0`` makes the column ``inf``, and NNLS has no answer for a
    design matrix with an infinity in it -- so a round-number seed silently
    destroyed the flux bootstrap for every instrument.
    """
    t = np.array([2458554.0 - 1.0, 2458554.0, 2458554.0 + 1.0])
    zeros = np.zeros_like(t)
    col = MulensInstrument._pspl_magnification(
        t, zeros, zeros, 2458554.0, 0.0, 20.0, 0.0, 0.0
    )
    assert np.all(np.isfinite(col))
    assert float(col[1]) == pytest.approx(1.0 / U_0_FLOOR, rel=1e-12)


def test_no_hard_coded_u_0_clip_survives_in_the_mulensing_sources():
    """
    Given the three modules that used to carry their own copy of the clip,
    When their sources are scanned,
    Then none of them spells it out again: the ``sign(...) * max(abs(...))``
      form appears nowhere, and every u_0 floor goes through the two shared
      helpers.  Three copies drifting apart is exactly how the 1e-6 / 1e-9
      disagreement happened; a fourth must not be able to appear silently.
    """
    for module in (lens_module, op_module, mi_module):
        src = inspect.getsource(module)
        code = "\n".join(
            line
            for line in src.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "np.sign(u0)" not in code and "np.sign(u_0)" not in code
        assert "pt.sign(u0" not in code and "pt.sign(u_0" not in code
