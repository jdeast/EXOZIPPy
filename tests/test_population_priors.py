"""Population-conditional age and [Fe/H] priors (review 1.7.5).

galacticmodel/populations.py is the ONE definition of each Galactic
population's age distribution and metallicity distribution function; the
present-day mass function marginalizes the ages, and a star whose age /
initfeh is sampled takes them inside the thin/thick/bulge mixture.  These
tests pin: each population's marginal against its literature inputs; that
the PDMF's alive fraction is exactly the shared age distribution's CDF;
finite C and JAX gradients of the mixture over the whole (age, [Fe/H], R)
range; and, through a System build with a synthetic MIST grid, that the
mixture marginal is the density-weighted mix of the per-population
distributions, that both booleans switch it, and that a star with no
sampled age is untouched.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager, write_synthetic_mist_grid
from exozippy.components.galacticmodel import galacticmodel as gm
from exozippy.components.galacticmodel import pdmf, populations
from exozippy.components.galacticmodel.populations import (
    AGE_MAX_GYR,
    POPULATIONS,
)


def _age_pdf(pop, t, lo=0.0, hi=AGE_MAX_GYR):
    return np.exp(
        populations.log_age_density(pop, t, xp=np)
        - populations.log_age_norm(pop, lo, hi)[0]
    )


def _moments(x, p):
    m = np.trapezoid(x * p, x)
    return m, np.sqrt(np.trapezoid((x - m) ** 2 * p, x))


def test_each_age_distribution_matches_its_inputs():
    """
    Given the three age distributions on [0, 13.77] Gyr,
    When each is normalized and its moments taken,
    Then each integrates to 1; the thin disk is exp(T/7 Gyr) on 0-10 Gyr
      (its analytic mean) to the 0.1 Gyr edge's precision; the thick disk
      is 12 +/- 1 truncated at 13.77; the bulge 9 +/- 1.
    """
    t = np.linspace(0.0, AGE_MAX_GYR, 200001)
    for pop in POPULATIONS:
        assert np.trapezoid(_age_pdf(pop, t), t) == pytest.approx(1, abs=1e-6)

    tau, tmax = 7.0, 10.0
    analytic = tmax / (1 - np.exp(-tmax / tau)) - tau  # mean of e^(T/tau)
    mean, _ = _moments(t, _age_pdf("thin_disk", t))
    assert mean == pytest.approx(analytic, abs=0.02)
    assert np.trapezoid(_age_pdf("thin_disk", t)[t > 10.5], t[t > 10.5]) < 1e-3

    from scipy.stats import truncnorm

    mean, sd = _moments(t, _age_pdf("thick_disk", t))
    ref = truncnorm((0 - 12) / 1.0, (AGE_MAX_GYR - 12) / 1.0, 12, 1.0)
    assert mean == pytest.approx(ref.mean(), abs=1e-4)
    assert sd == pytest.approx(ref.std(), abs=1e-4)
    mean, sd = _moments(t, _age_pdf("bulge", t))
    assert mean == pytest.approx(9.0, abs=1e-4)
    assert sd == pytest.approx(1.0, abs=1e-4)


def test_each_mdf_matches_its_inputs():
    """
    Given the three metallicity distribution functions,
    When their moments are taken on the real line,
    Then the bulge is Hill+2011's 45/55 mixture (mean +0.041, the analytic
      variance), the thick disk -0.78 +/- 0.30, and the thin disk is
      0.2 dex wide, solar at R0, +0.17 in the inner disk and -0.41 beyond
      15 kpc (-0.06 dex/kpc between).
    """
    f = np.linspace(-6, 6, 240001)
    (w1, m1, s1), (w2, m2, s2) = populations.BULGE_FEH_COMPONENTS
    p = np.exp(populations.log_feh_density("bulge", f, xp=np))
    assert np.trapezoid(p, f) == pytest.approx(1, abs=1e-9)
    mean, sd = _moments(f, p)
    assert mean == pytest.approx(w1 * m1 + w2 * m2, abs=1e-9)
    var = w1 * (s1**2 + m1**2) + w2 * (s2**2 + m2**2) - mean**2
    assert sd == pytest.approx(np.sqrt(var), abs=1e-9)
    assert populations.POPULATION_FEH["bulge"] == pytest.approx(0.041)

    p = np.exp(populations.log_feh_density("thick_disk", f, xp=np))
    assert _moments(f, p) == pytest.approx((-0.78, 0.30), abs=1e-9)

    for r, center in (
        (8.16, 0.0),
        (1.0, 0.1716),
        (12.16, -0.24),
        (25, -0.4104),
    ):
        p = np.exp(populations.log_feh_density("thin_disk", f, r, xp=np))
        assert _moments(f, p) == pytest.approx((center, 0.2), abs=2e-3), r


def test_pdmf_alive_fraction_is_the_shared_age_cdf():
    """
    Given the PDMF's age-marginalized alive fraction (review 1.7.2),
    When it is compared with a direct quadrature of the SAME age
      distribution the per-star prior uses, up to each mass's lifetime,
    Then they agree for every population -- one definition, two consumers.
    """
    logm = np.linspace(-0.2, 0.6, 41)
    t = np.linspace(0.0, AGE_MAX_GYR, 400001)
    for pop in POPULATIONS:
        tau = 10 ** pdmf.log10_lifetime_gyr(
            logm, populations.POPULATION_FEH[pop]
        )
        pdf = _age_pdf(pop, t)
        direct = [np.trapezoid(pdf[t <= tt], t[t <= tt]) for tt in tau]
        assert pdmf.alive_fraction(pop, logm) == pytest.approx(
            direct, abs=2e-4
        ), pop


def test_thick_disk_turnoff_matches_genulens():
    """
    Given the thick disk's turnoff metallicity, its MDF mean (-0.78),
    When its 12 Gyr dying mass is read off the MIST lifetimes,
    Then it is genulens's Minidie.dat value, 0.857 Msun, to 0.01.
    """
    from scipy.optimize import brentq

    lm = brentq(
        lambda x: (
            pdmf.log10_lifetime_gyr(
                x, populations.POPULATION_FEH["thick_disk"]
            )
            - np.log10(12.0)
        ),
        -0.3,
        0.3,
    )
    assert 10**lm == pytest.approx(0.857, abs=0.01)


def _mixture(t, f, r):
    """The three branches' age + [Fe/H] terms, mixed with equal weights,
    as the model writes them (normalized over [0, 13.77] and [-2.5, 0.5])."""
    lo_f, hi_f = -2.5, 0.5
    terms = []
    for pop in POPULATIONS:
        a = populations.log_age_density(pop, t) - populations.log_age_norm(
            pop, 0.0, AGE_MAX_GYR
        )
        b = populations.log_feh_density(pop, f, r_kpc=r)
        if pop == "thin_disk":
            b = b - populations.thin_disk_log_feh_norm(r, lo_f, hi_f)
        else:
            b = b - populations.log_feh_norm(pop, lo_f, hi_f)
        terms.append(a + b)
    return pt.logsumexp(pt.stack(terms), axis=0)


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_mixture_gradient_finite_everywhere(mode):
    """
    Given the age/[Fe/H] mixture over the three populations,
    When its value and gradient are evaluated over the whole support
      (ages 0-13.77 Gyr through the thin disk's 10 Gyr edge, [Fe/H] -2.5
      to +0.5, R from the Galactic center through both gradient hinges to
      30 kpc),
    Then every value and gradient is finite on the C and the JAX backends.
    """
    if mode == "JAX":
        pytest.importorskip("jax")
    tv, fv, rv = pt.dvector("t"), pt.dvector("f"), pt.dvector("r")
    lp = _mixture(tv, fv, rv)
    g = pytensor.grad(pt.sum(lp), [tv, fv, rv])
    fn = pytensor.function([tv, fv, rv], [lp, *g], mode=mode)
    t, f, r = np.meshgrid(
        np.concatenate([[0.0], np.linspace(9.5, 10.5, 21), [13.77]]),
        np.array([-2.5, -0.78, 0.0, 0.32, 0.5]),
        np.array([0.0, 1e-3, 5.3, 8.16, 15.0, 30.0]),
        indexing="ij",
    )
    out = fn(t.ravel(), f.ravel(), r.ravel())
    for arr in out:
        assert np.all(np.isfinite(np.asarray(arr)))


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_bar_density_gradient_finite_on_its_axes(mode):
    """
    Given K21's E+E_X bar density (review 1.7.3), whose boxy radius has
      |x|^C1 terms that a naive abs would make singular on the bar's axes,
    When the three branch log densities and their gradient are evaluated on
      a grid through the exact Galactic center, the bar's axes, the
      X-shape's shear planes and both cylindrical cutoffs,
    Then every value and gradient is finite on the C and JAX backends.
    """
    if mode == "JAX":
        pytest.importorskip("jax")
    xv, yv, zv = pt.dvector("x"), pt.dvector("y"), pt.dvector("z")
    r = pt.sqrt(xv**2 + yv**2 + 1e-6)
    lp = pt.logsumexp(
        pt.stack(gm.branch_log_number_densities(xv, yv, zv, r)), axis=0
    )
    g = pytensor.grad(pt.sum(lp), [xv, yv, zv])
    fn = pytensor.function([xv, yv, zv], [lp, *g], mode=mode)
    th = np.radians(27.0)
    s = np.array([-3.0, -1.3, -0.5, -1e-9, 0.0, 1e-9, 0.5, 1.3, 2.7, 8.0])
    xb, yb, z = np.meshgrid(s, s, np.array([-0.5, -1e-9, 0.0, 0.3]))
    x = xb * np.cos(th) - yb * np.sin(th)
    y = xb * np.sin(th) + yb * np.cos(th)
    for arr in fn(x.ravel(), y.ravel(), z.ravel()):
        assert np.all(np.isfinite(np.asarray(arr)))


def test_thin_disk_support_check_refuses_a_support_the_mdf_misses():
    """
    Given an initfeh support far above every thin-disk center,
    When the support is checked,
    Then it raises naming the parameter (the model used outside its
      domain), and the MIST grid's [-2.5, 0.5] passes.
    """
    populations.check_thin_disk_feh_support([-2.5], [0.5], "star.initfeh")
    with pytest.raises(ValueError, match="star.initfeh"):
        populations.check_thin_disk_feh_support([2.0], [3.0], "star.initfeh")


@pytest.mark.parametrize("key", ["population_age", "population_feh"])
def test_population_switches_must_be_booleans(key):
    """
    Given a population switch spelled as a string,
    When the galactic model is constructed,
    Then it raises rather than reading "false" as true.
    """
    with pytest.raises(ValueError, match=f"{key} must be true or false"):
        gm.GalacticModel([{key: "false"}], _DummyConfigManager())


# ----------------------------------------------------------------------
# Through a System: a MIST star on a bulge sight line.
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    return write_synthetic_mist_grid(tmp_path_factory.mktemp("mist_pop"))


def _system(model_root, mist=True, **gm_keys):
    # present_day off unless asked: with it on, a star whose age is sampled
    # also swaps its age-marginalized PDMF term for the IMF (the luminous
    # joint), which is not what the marginal tests isolate.
    gm_keys.setdefault("present_day", False)
    # the 3-population priors these tests pin are model: simplified's
    gm_keys.setdefault("model", "simplified")
    from exozippy.system import System

    config = {
        "star": [{"name": "S", "mist": mist}],
        "galacticmodel": [{"name": "gm", **gm_keys}],
        "logger_level": "WARNING",
    }
    if mist:
        config["evolutionarymodel"] = [{"star": "S", "model_root": model_root}]
    params = {
        "star.S.ra": {"initval": 270.9, "sigma": 0.0},  # Baade's window
        "star.S.dec": {"initval": -30.03, "sigma": 0.0},
        "star.S.distance": {"initval": 8000.0, "lower": 1.0, "upper": 20000},
        "star.S.pm_ra": {"initval": -3.0, "lower": -100, "upper": 100},
        "star.S.pm_dec": {"initval": -5.0, "lower": -100, "upper": 100},
        "star.S.rv": {"initval": 0.0},
        "star.S.mass": {"initval": 1.0},
    }
    system = System(config, params)
    system.prepare()
    return system, system.build_model()


def _kin_vs(system, model, name, values):
    """kinematic_prior at the start with star.<name> replaced by each of
    ``values`` (the potential as a function of that one coordinate)."""
    from pytensor.graph.replace import graph_replace

    # Swap the coordinate out, THEN put value variables in place of the
    # remaining random variables: compiling a graph that still holds an RV
    # draws from it instead of reading the start.
    param = getattr(system.star, name)
    x = pt.dvector("x")
    expr = graph_replace(
        model["galacticmodel.kinematic_prior"], {param.value: x}
    )
    (expr,) = model.replace_rvs_by_values([expr])
    fn = model.compile_fn(
        expr, inputs=[*model.value_vars, x], on_unused_input="ignore"
    )
    start = system.get_raw_start(model)
    return np.array([fn({**start, "x": np.array([v])}) for v in values])


@pytest.mark.slow
def test_mixture_marginal_is_the_density_weighted_mix(model_root):
    """
    Given a MIST star at 8 kpc toward Baade's window, under the galactic
      model with only the age (then only the [Fe/H]) prior on,
    When the kinematic potential is scanned in that coordinate,
    Then exp(it - the prior-off potential) is sum_b r_b g_b(T) with
      responsibilities r_b >= 0 that sum to 1 (solved from three ages and
      checked at the rest -- the branch weights are the model's own
      density x kinematics), the bulge carries most of the weight, and the
      marginal integrates to 1 over the support.
    """
    off_sys, off = _system(
        model_root, population_age=False, population_feh=False
    )
    base = _kin_vs(off_sys, off, "distance", [8000.0])[0]

    sys_a, m_a = _system(model_root, population_feh=False)
    lo_t, hi_t = (
        float(np.atleast_1d(b)[0])
        for b in (sys_a.star.age.lower, sys_a.star.age.upper)
    )
    t = np.linspace(lo_t, hi_t, 1401)
    dens = np.exp(_kin_vs(sys_a, m_a, "age", t) - base)
    assert np.trapezoid(dens, t) == pytest.approx(1.0, abs=1e-4)
    g = np.stack([_age_pdf(pop, t, lo_t, hi_t) for pop in POPULATIONS], axis=1)
    resp, *_ = np.linalg.lstsq(g, dens, rcond=None)
    assert np.all(resp > -1e-6) and resp.sum() == pytest.approx(1, abs=1e-5)
    assert np.max(np.abs(g @ resp - dens)) < 1e-6 * dens.max() + 1e-9
    assert resp[2] > 0.5  # bulge-dominated at 8 kpc

    sys_f, m_f = _system(model_root, population_age=False)
    lo, hi = (
        float(np.atleast_1d(b)[0])
        for b in (sys_f.star.initfeh.lower, sys_f.star.initfeh.upper)
    )
    f = np.linspace(lo, hi, 1201)
    dens_f = np.exp(_kin_vs(sys_f, m_f, "initfeh", f) - base)
    assert np.trapezoid(dens_f, f) == pytest.approx(1.0, abs=1e-4)
    # The same responsibilities mix the (truncated, renormalized) MDFs.
    r8 = float(
        np.hypot(*_star_xy(sys_f))  # the star's Galactocentric R
    )
    mdf = []
    for pop in POPULATIONS:
        p = np.exp(populations.log_feh_density(pop, f, r8, xp=np))
        mdf.append(p / np.trapezoid(p, f))
    assert (
        np.max(np.abs(np.stack(mdf, 1) @ resp - dens_f)) < 1e-3 * dens_f.max()
    )


def _star_xy(system):
    from exozippy.components.galacticmodel.physics import (
        galactic_xyz,
        line_of_sight_basis,
    )

    st = system.star
    _, _, sun, los = line_of_sight_basis(
        st.ra.element_start(0), st.dec.element_start(0)
    )
    x, y, _ = galactic_xyz(st.distance.element_start(0) / 1e3, sun, los)
    return x, y


@pytest.mark.slow
def test_a_star_without_a_track_is_untouched(model_root):
    """
    Given the same star with no evolutionary track (no sampled age or
      initfeh),
    When the model is built with the population priors on and off,
    Then the kinematic potential is bit-identical, and with a track the
      prior is declared to the Prior column on age and initfeh.
    """
    on_sys, on = _system(model_root, mist=False)
    off_sys, off = _system(
        model_root, mist=False, population_age=False, population_feh=False
    )
    v_on = _kin_vs(on_sys, on, "distance", [8000.0])
    v_off = _kin_vs(off_sys, off, "distance", [8000.0])
    assert v_on[0] == v_off[0]

    sys_t, _ = _system(model_root)
    for name in ("age", "initfeh"):
        texts = [c.text for c in getattr(sys_t.star, name).prior_contributions]
        assert any("Galactic-population mixture" in t for t in texts), name
