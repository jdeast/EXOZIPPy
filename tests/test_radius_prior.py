"""The planet-radius base prior: uniform in p = R_P/R_*, not in R_P.

EXOFASTv2 steps in p with a flat prior; EXOZIPPy samples `planet.radius`,
whose logit transform is exactly uniform in R_P.  Uniform in R_P is R_* times
uniform in p on the joint (p, R_*) density, which tilts a transit fit toward a
larger star.  `Planet._add_radius_prior` adds `-log R_*` of each SAMPLED
planet's host so the implied prior is flat in (p, R_*) -- checked here as a
property, carried from the raw coordinates through the model's own transform,
in the style of tests/test_fitmsini.py.

See src/exozippy/components/star/star.md, "Planet radius prior".
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.planet import physics
from exozippy.system import System

JACOBIAN = "planet.radius_jacobian"


def _values(model, nodes):
    """`nodes` rewritten onto the model's VALUE variables (compiling a
    Parameter's `.value` against `value_vars` directly evaluates a prior
    draw, not the point passed in)."""
    return model.replace_rvs_by_values(list(nodes))


def _build(radius_entries, n_planets=1, star_radius=None):
    """No data; everything pinned but the planet radii and the star radius
    (unless the caller pins those too)."""
    names = ["b", "c"][:n_planets]
    config = {
        "star": [{"name": "A", "mist": False}],
        # fitlogq: false -> planet.mass is the SAMPLED coordinate, so its
        # sigma: 0 pin holds (a pin on a derived mass raises, 2.1.26(e)).
        "planet": [{"name": nm, "fitlogq": False} for nm in names],
        "orbit": [
            {"name": nm, "primary": ["A"], "companion": [nm]} for nm in names
        ],
    }
    params = {
        "star.A.logmass": {"initval": 0.0, "sigma": 0.0},
        "star.A.radius": star_radius or {"initval": 1.0},
    }
    for i, (nm, rad) in enumerate(zip(names, radius_entries)):
        params.update(
            {
                f"orbit.{nm}.logP": {"initval": 0.5 + i, "sigma": 0.0},
                f"orbit.{nm}.tc": {"initval": 2455000.0, "sigma": 0.0},
                f"orbit.{nm}.secosw": {"initval": 0.0, "sigma": 0.0},
                f"orbit.{nm}.sesinw": {"initval": 0.0, "sigma": 0.0},
                f"orbit.{nm}.cosi": {"initval": 0.0, "sigma": 0.0},
                f"planet.{nm}.mass": {"initval": 1.0, "sigma": 0.0},
                f"planet.{nm}.radius": rad,
            }
        )
    system = System(config, params)
    system.prepare()
    return system, system.build_model()


@pytest.fixture(scope="module")
def free_model():
    return _build([{"initval": 1.0}])


@pytest.fixture(scope="module")
def pinned_model():
    return _build([{"initval": 1.0, "sigma": 0.0}])


def _jacobian_potentials(model):
    return [p for p in model.potentials if p.name == JACOBIAN]


def test_a_sampled_radius_gets_the_jacobian(free_model):
    system, model = free_model
    free = {v.name for v in model.free_RVs}
    assert free == {"planet.radius_raw", "star.radius_raw"}
    assert system.planet.chen == [False]  # no data: nothing else on R_P
    assert len(_jacobian_potentials(model)) == 1


def test_a_pinned_radius_gets_no_jacobian(pinned_model):
    """
    Given `planet.radius: {sigma: 0}`,
    When the model is built,
    Then no radius Jacobian is added -- it would be a function of R_* alone,
      a spurious prior on the star -- and the Prior column still says Fixed.
    """
    system, model = pinned_model
    assert "planet.radius_raw" not in {v.name for v in model.free_RVs}
    assert _jacobian_potentials(model) == []
    assert system.planet.radius.prior_contributions_at(0) == []
    assert "Fixed" in system.planet.radius.get_prior_str(0, latex=False)


def test_only_the_sampled_planet_of_a_mixed_pair_gets_the_term():
    """One planet sampled, one pinned: the potential is -log R_* exactly
    once (both planets orbit the one star)."""
    system, model = _build(
        [{"initval": 1.0}, {"initval": 0.5, "sigma": 0.0}], n_planets=2
    )
    pots = _jacobian_potentials(model)
    assert len(pots) == 1
    assert system.planet.radius.is_sampled.tolist() == [True, False]
    f = pytensor.function(
        model.value_vars,
        _values(model, [pots[0], system.star.radius.value[0]]),
        on_unused_input="ignore",
    )
    start = model.initial_point()
    jt, rstar = f(*[start[v.name] for v in model.value_vars])
    assert float(jt) == pytest.approx(-np.log(float(rstar)), rel=1e-12)
    assert system.planet.radius.prior_contributions_at(1) == []
    assert system.planet.radius.prior_contributions_at(0) != []


def _physical(system, model):
    """(logp, p, R_*, log|det d(p, R_*)/d(raw_rp, raw_rs)|, jacobian term)
    as functions of the two free raw coordinates."""
    vv = {v.name: v for v in model.value_vars}
    rraw, sraw = vv["planet.radius_raw"], vv["star.radius_raw"]
    p, rstar = _values(
        model, [system.planet.p.value[0], system.star.radius.value[0]]
    )

    def d(y, x):
        return pytensor.grad(y, x, disconnected_inputs="ignore")[0]

    det = d(p, rraw) * d(rstar, sraw) - d(p, sraw) * d(rstar, rraw)
    jt = _values(model, _jacobian_potentials(model))[0]
    outs = [model.logp(), p, rstar, pt.log(pt.abs(det)), jt]
    return pytensor.function([rraw, sraw], outs, on_unused_input="ignore")


def test_the_implied_prior_is_flat_in_p_not_in_rp(free_model):
    """
    Given no likelihood, everything but R_P and R_* pinned,
    When the model's density is carried from its raw coordinates to
      (p, R_*) -- log p(p, R_*) = logp(raw) - log|det J|,
    Then it is constant: the prior is uniform in p = R_P/R_* (EXOFASTv2's).
      And without the Jacobian term it is NOT flat in p -- what is left is
      exactly +log R_*, i.e. the old prior uniform in R_P, which favors a
      larger star.  The sign is checked as a property, not argued.
    """
    system, model = free_model
    f = _physical(system, model)
    dens, nojac, rstars = [], [], []
    for r in np.linspace(-3.0, 3.0, 7):
        for s in np.linspace(-15.0, 15.0, 7):
            lp, p, rstar, logdet, jt = f(np.array([r]), np.array([s]))
            assert 0.0 < p < 1.0
            dens.append(lp - logdet)
            nojac.append(lp - jt - logdet)
            rstars.append(rstar)
    rstars = np.array(rstars)
    # The grid spans a real range of R_*, or flatness would prove nothing.
    assert rstars.max() / rstars.min() > 1.5
    dens = np.array(dens)
    assert np.ptp(dens) < 1e-6, np.ptp(dens)
    nojac = np.array(nojac)
    assert np.ptp(nojac) > 0.3
    assert np.ptp(nojac - np.log(rstars)) < 1e-6


def test_the_model_gradient_is_exact(free_model):
    """dlogp agrees with finite differences of logp, so the floored log
    leaves the gradient exact for every physical radius."""
    _, model = free_model
    logp = model.compile_logp()
    dlogp = model.compile_dlogp()
    point = {
        "planet.radius_raw": np.array([0.7]),
        "star.radius_raw": np.array([-2.0]),
    }
    grad = np.ravel(dlogp(point))
    for k, name in enumerate(v.name for v in model.value_vars):
        e = 1e-6
        hi, lo = dict(point), dict(point)
        hi[name] = point[name] + e
        lo[name] = point[name] - e
        fd = (logp(hi) - logp(lo)) / (2 * e)
        assert grad[k] == pytest.approx(fd, rel=1e-5, abs=1e-7), name


def test_the_floor_keeps_the_term_finite_at_a_nonpositive_radius():
    """A user may widen star.radius to 0 or drop its bounds; the strictly
    positive floor inside the log keeps the term and its gradient finite
    (the CLAUDE.md where-trap rule)."""
    r = pt.dscalar("r")
    j = physics.radius_base_prior_logp(r)
    f = pytensor.function([r], [j, pt.grad(j, r)])
    for edge in (0.0, -1.0, 1e-300):
        assert all(np.isfinite(v) for v in f(edge))
    # ...and a physical radius is untouched.
    assert f(1.3)[0] == pytest.approx(-np.log(1.3), rel=1e-15)


def test_the_prior_column_and_the_prose_say_uniform_in_p(free_model):
    system, _ = free_model
    txt = system.planet.radius.get_prior_str(0, latex=False)
    assert "Rp/R*" in txt
    cell, notes = system.planet.radius.prior_cell_and_notes(0)
    assert any(r"1/R_*" in n for n in notes)
    prose = " ".join(
        s.text for s in system.prose.sentences() if "radius_prior" in s.key
    )
    assert r"uniform in $R_P/R_*$" in prose
    assert "Eastman:2019" in prose


def test_a_user_gaussian_keeps_the_jacobian_and_both_priors_are_reported():
    """
    Given a Gaussian user prior on R_P,
    When the model is built,
    Then the Jacobian is still added (the Gaussian multiplies the base
      prior, as EXOFASTv2's rp penalty multiplies its flat-in-p stepping),
      and the Prior column reports both the Gaussian and the 1/R_* term.
    """
    system, model = _build([{"initval": 1.0, "mu": 1.0, "sigma": 0.1}])
    assert len(_jacobian_potentials(model)) == 1
    txt = system.planet.radius.get_prior_str(0, latex=False)
    assert "Rp/R*" in txt
    assert "1" in txt and "0.1" in txt
