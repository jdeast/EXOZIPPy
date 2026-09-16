"""An exactly circular start must not make the gradient NaN (review 1.8.2).

`secosw: 0` + `sesinw: 0` is how a params file says "start circular", and it
is the one seed that puts BOTH arctan2 arguments of the Tc -> Tp inversion at
the origin and drives `d(sqrt e)/de` to infinity.  Three of the tests below
fail pre-fix (`calc_tp`, `calc_esinw`, `calc_ecosw`) and so does the
end-to-end one; the two angle tests pass either way and pin a hardening --
see `physics._circular_bias`.  Before the fix the model
built, its logp was finite, and only the GRADIENT was NaN -- so NUTS
initialization, the whitening probe and the gradient polish all started on a
NaN with nothing naming the cause.  Measured end to end on
`examples/kelt4/kelt4_rvonly.yaml` with those two seeds added: start logp
-600.9969443376475 either way, `dlogp` with exactly two non-finite entries
before (`orbit.secosw_raw`, `orbit.sesinw_raw`) and none after, every other
entry bit-identical.

The shields must also be INERT: `pt.maximum(e, 1e-30)` returns its argument
bit-for-bit above the floor and `_circular_bias` adds exactly `0.0` off the
origin, so no non-circular fit may move by a single ulp.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.orbit import Orbit, physics

# Use pt.dscalar, never a bare python float: pytensor autocasts a bare float
# to the smallest dtype that represents it, and a unary op on it then computes
# in float32 (docs/testing.md).
SECOSW = pt.dscalar("secosw")
SESINW = pt.dscalar("sesinw")


def _value_and_grad(node, inputs, point):
    """(value, [d node / d input]) evaluated at `point`."""
    grads = pytensor.grad(pt.sum(node), inputs, disconnected_inputs="ignore")
    fn = pytensor.function(inputs, [node] + list(grads))
    out = fn(*point)
    return out[0], np.asarray(out[1:], dtype=float)


# ---------------------------------------------------------------------------
# The individual physics functions
# ---------------------------------------------------------------------------


def test_calc_omega_is_finite_and_differentiable_at_exactly_zero():
    """
    Given the sqrt(e) pair at exactly (0, 0), where the argument of
      periastron is undefined and the house convention is 90 deg,
    When calc_omega and its gradient are evaluated there,
    Then the value is exactly pi/2 and the gradient is finite.  This one
      PASSES pre-fix -- a rewrite defuses the switch on the C backend -- and
      is pinned so the hardening cannot be dropped as unnecessary if that
      rewrite ever stops firing.
    """
    # Arrange / Act
    val, grad = _value_and_grad(
        physics.calc_omega(SECOSW, SESINW), [SECOSW, SESINW], (0.0, 0.0)
    )

    # Assert
    assert val == pytest.approx(np.pi / 2.0, abs=0.0)
    assert np.isfinite(grad).all()


def test_calc_tp_is_finite_and_differentiable_at_exactly_zero():
    """
    Given an exactly circular orbit, where calc_tp's two arctan2 arguments
      vanish together and d(sqrt e)/de is infinite,
    When tp and its gradient with respect to the sqrt(e) pair are evaluated,
    Then tp is the time of conjunction itself (the e = 0, omega = 90 deg
      orbit calc_omega's convention names) and the gradient is finite.
    """
    tc, n = 2460000.0, 2.0 * np.pi / 10.0
    ecc = physics.calc_ecc(SECOSW, SESINW)
    node = physics.calc_tp(ecc, SESINW, SECOSW, tc, n)

    val, grad = _value_and_grad(node, [SECOSW, SESINW], (0.0, 0.0))

    assert val == pytest.approx(tc, abs=1e-9)
    assert np.isfinite(grad).all()


@pytest.mark.parametrize("func", ["calc_esinw", "calc_ecosw"])
def test_the_linear_e_vector_is_differentiable_at_exactly_zero(func):
    """
    Given e = 0, where `sqrt(e)`'s derivative is infinite and
      `de/d(secosw) = 2 secosw` is exactly zero, so the chain rule
      multiplies inf by 0,
    When e sin(omega) / e cos(omega) and their gradients are evaluated,
    Then both are zero with a finite gradient.
    """
    ecc = physics.calc_ecc(SECOSW, SESINW)
    pair = SESINW if func == "calc_esinw" else SECOSW
    node = getattr(physics, func)(ecc, pair)

    val, grad = _value_and_grad(node, [SECOSW, SESINW], (0.0, 0.0))

    assert val == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(grad).all()


def test_calc_lam_from_sv_is_finite_and_differentiable_at_exactly_zero():
    """
    Given an exactly aligned start -- `svcoslam: 0, svsinlam: 0`, the
      Rossiter-McLaughlin twin of the circular seed,
    When the spin-orbit angle and its gradient are evaluated,
    Then the value is 0 (aligned) and the gradient is finite.  Like
      calc_omega above this passes pre-fix and pins the hardening.
    """
    val, grad = _value_and_grad(
        physics.calc_lam_from_sv(SECOSW, SESINW), [SECOSW, SESINW], (0.0, 0.0)
    )

    assert val == pytest.approx(0.0, abs=0.0)
    assert np.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Inertness away from the origin
# ---------------------------------------------------------------------------


_OFF_ORIGIN = [
    (0.3, 0.2),
    (-0.4, 0.1),
    (0.0, 0.5),
    (0.5, 0.0),
    (-0.1, -0.7),
    (1e-8, 1e-8),
]


def _unshielded(ecc, secosw, sesinw, tc, n):
    """The four expressions exactly as they read before review 1.8.2.

    Built in PyTensor, not transcribed into numpy: the claim is that the
    shields are inert in the graph the model compiles, and a numpy
    transcription answers a different (and, at the last bit, differently
    rounded) question.
    """
    e0 = 2.0 * pt.arctan2(
        pt.sqrt(1.0 - ecc) * (pt.sqrt(ecc) - sesinw),
        pt.sqrt(1.0 + ecc) * secosw,
    )
    return [
        pt.switch(
            pt.eq(pt.sqr(sesinw) + pt.sqr(secosw), 0.0),
            np.pi / 2.0,
            pt.arctan2(sesinw, secosw),
        ),
        tc - (e0 - ecc * pt.sin(e0)) / n,
        pt.sqrt(ecc) * sesinw,
        pt.sqrt(ecc) * secosw,
    ]


@pytest.mark.parametrize("secosw,sesinw", _OFF_ORIGIN)
def test_the_shields_are_bit_identical_off_the_origin(secosw, sesinw):
    """
    Given any sqrt(e) pair that is not exactly (0, 0),
    When omega, tp, e sin(omega) and e cos(omega) are evaluated with and
      without the shields,
    Then every one agrees to the last bit -- the floor returns its argument
      unchanged above 1e-30 and the circular bias adds exactly 0.0, so no
      shipped fit may move.
    """
    # Arrange
    tc, n = 2460000.0, 2.0 * np.pi / 10.0
    ecc = physics.calc_ecc(SECOSW, SESINW)
    shielded = [
        physics.calc_omega(SECOSW, SESINW),
        physics.calc_tp(ecc, SESINW, SECOSW, tc, n),
        physics.calc_esinw(ecc, SESINW),
        physics.calc_ecosw(ecc, SECOSW),
    ]
    plain = _unshielded(ecc, SECOSW, SESINW, tc, n)

    # Act
    out = pytensor.function([SECOSW, SESINW], shielded + plain)(secosw, sesinw)

    # Assert -- exact equality, not approx.
    for name, got, want in zip(
        ("omega", "tp", "esinw", "ecosw"), out[:4], out[4:]
    ):
        assert got == want, f"{name} moved: {got!r} != {want!r}"


# ---------------------------------------------------------------------------
# The whole model, through the raw coordinates
# ---------------------------------------------------------------------------


def _write_rv(path, n=40):
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, n))
    rv = 30.0 * np.sin(2 * np.pi * t / 17.0) + rng.normal(0, 3.0, n)
    np.savetxt(path, np.column_stack([t, rv, np.full(n, 3.0)]))
    return str(path)


@pytest.fixture(scope="module")
def rv_file(tmp_path_factory):
    return _write_rv(tmp_path_factory.mktemp("circular_rv") / "a.rv")


def test_an_exactly_circular_params_file_keeps_a_finite_start_gradient(
    rv_file,
):
    """
    Given an RV fit whose params file says `secosw: 0` and `sesinw: 0` --
      the way a user spells "start this orbit circular",
    When the model is built and its logp and gradient are evaluated at the
      start point,
    Then both are finite.  Before the fix the logp alone was finite and the
      two sqrt(e) entries of the gradient were NaN, so NUTS initialization,
      the whitening probe and the gradient polish all began on a NaN with
      no error naming the parameter responsible.
    """
    # Arrange
    from exozippy.system import System

    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": [{"name": "inst", "file": rv_file}],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.logP": {"initval": np.log10(17.0)},
        "orbit.b.tc": {"initval": 2455010.0},
        "orbit.b.secosw": {"initval": 0.0},
        "orbit.b.sesinw": {"initval": 0.0},
    }
    system = System(config, params)
    system.prepare()
    model = system.build_model()

    # Act
    point = model.initial_point()
    logp = model.compile_logp()(point)
    grads = model.compile_dlogp()(point)
    flat = np.concatenate([np.atleast_1d(g) for g in np.atleast_1d(grads)])

    # Assert
    assert np.isfinite(logp)
    assert np.isfinite(flat).all(), f"non-finite gradient entries: {flat}"
