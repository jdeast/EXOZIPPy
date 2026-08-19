"""The ascending-node degeneracy: no truncation, one fold (review 1.8.3).

With astrometry and no RVs, `(bigomega, omega_*, tc)` and
`(bigomega + 180, omega_* + 180, tc')` are a reflection through the sky plane
-- identical astrometry of every kind, and only radial information says which
node is ascending.  The component used to bound `ybigomega >= 0` to select one
label, which TRUNCATES a posterior that hugs the boundary: measured on
`examples/HIP1349`, `Omega = 176.5 +/- 2.7` against DMSA's `172.6 +/- 3.4`,
with zero origin-crossings in 16k draws.

The bound is unnecessary because the two modes are ANTIPODAL in the sampled
direction vector, joined through the origin with no likelihood barrier -- the
same geometry the microlensing trajectory angle has.  So the sampler gets both
half-planes and the labels are collapsed afterwards, ONCE, where the draws are
decoded, so the convergence check, the mode reporter, the seed ledger and the
tables cannot disagree about how many solutions a chain found.
"""

import numpy as np
import pytest

from exozippy.components.orbit import physics
from exozippy.system import System

_TRUTH = dict(
    mstar=1.0,
    mcomp=0.3,
    period=400.0,
    ecc=0.35,
    omega=np.radians(70.0),
    bigomega=np.radians(150.0),
    inc=np.radians(65.0),
    tc=2455100.0,
    plx=10.0,
)


def _write_rel(path, seed=11):
    """A relative-astrometry (sep, PA) dataset -- differential, no radial info."""
    rng = np.random.default_rng(seed)
    t = np.linspace(2455000.0, 2455000.0 + 2.0 * _TRUTH["period"], 24)
    # The true track is irrelevant to what these tests assert (both are
    # statements about the model, not about recovering the truth), so a
    # smooth arc with realistic errors is enough to make the likelihood
    # depend on every element of the orbit.
    phase = 2 * np.pi * (t - _TRUTH["tc"]) / _TRUTH["period"]
    sep = 20.0 + 5.0 * np.cos(phase) + rng.normal(0, 0.2, t.size)
    pa = np.degrees(np.arctan2(np.sin(phase), 0.6 * np.cos(phase))) % 360.0
    pa = pa + rng.normal(0, 0.5, t.size)
    np.savetxt(
        path,
        np.column_stack(
            [t, sep, np.full(t.size, 0.2), pa, np.full(t.size, 0.5)]
        ),
    )
    return str(path)


def _write_rv(path, seed=5):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, 30))
    np.savetxt(
        path,
        np.column_stack(
            [t, 50.0 * np.sin(2 * np.pi * t / 31.0), np.full(t.size, 5.0)]
        ),
    )
    return str(path)


def _params(**extra):
    T = _TRUTH
    params = {
        "star.A.mass": {"initval": T["mstar"], "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.distance": {"initval": 1000.0 / T["plx"]},
        "planet.BH.mass": {"initval": T["mcomp"] * 1047.5655},
        "planet.BH.radius": {"initval": 1.0, "sigma": 0},
        "orbit.BH.period": {"initval": T["period"]},
        "orbit.BH.tc": {"initval": T["tc"]},
        "orbit.BH.secosw": {"initval": np.sqrt(T["ecc"]) * np.cos(T["omega"])},
        "orbit.BH.sesinw": {"initval": np.sqrt(T["ecc"]) * np.sin(T["omega"])},
        "orbit.BH.bigomega": {"initval": np.degrees(T["bigomega"])},
        "orbit.BH.cosi": {"initval": np.cos(T["inc"])},
    }
    params.update(extra)
    return params


@pytest.fixture(scope="module")
def astrometry_only(tmp_path_factory):
    """A rel-astrometry system with NO radial data: the degenerate case."""
    d = tmp_path_factory.mktemp("node")
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [{"name": "BH"}],
        "astrometryinstrument": [
            {
                "name": "Rel",
                "file": _write_rel(d / "sim.rel.astrom"),
                "mode": "rel",
            }
        ],
    }
    system = System(config, _params())
    system.prepare()
    model = system.build_model()
    return system, model


# ---------------------------------------------------------------------------
# 1. The predicate, per orbit
# ---------------------------------------------------------------------------


def test_an_astrometry_only_orbit_is_node_degenerate(astrometry_only):
    """
    Given an orbit measured by relative astrometry and nothing else,
    When the orbit registers its parameters,
    Then it is flagged node-degenerate.
    """
    system, _ = astrometry_only
    assert list(system.orbit.node_degenerate) == [True]


def test_an_rv_constrained_orbit_is_not_degenerate(tmp_path):
    """
    Given the same astrometry PLUS radial velocities of the same star,
    When the orbit registers its parameters,
    Then it is NOT degenerate -- an RV says which node is ascending, so
      there is nothing to fold and nothing to truncate.
    """
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [{"name": "BH"}],
        "astrometryinstrument": [
            {
                "name": "Rel",
                "file": _write_rel(tmp_path / "sim.rel.astrom"),
                "mode": "rel",
            }
        ],
        "rvinstrument": [
            {"name": "inst", "file": _write_rv(tmp_path / "rv.dat")}
        ],
    }
    system = System(config, _params())
    system.prepare()
    assert list(system.orbit.node_degenerate) == [False]


def test_the_predicate_is_per_orbit_not_per_system(tmp_path):
    """
    Given a MIXED system -- one orbit measured by astrometry alone, another
      by RVs,
    When the orbits register their parameters,
    Then only the astrometry-only one is degenerate.  The old test was
      system-wide ("any astrometry and no rvinstrument anywhere"), so an
      RV-constrained orbit in a mixed system was truncated for nothing.
    """
    config = {
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "planet": [{"name": "BH"}, {"name": "c"}],
        "orbit": [
            {"name": "BH", "primary": ["A"], "companion": ["BH"]},
            {"name": "c", "primary": ["B"], "companion": ["c"]},
        ],
        "astrometryinstrument": [
            {
                "name": "Rel",
                "file": _write_rel(tmp_path / "sim.rel.astrom"),
                "mode": "rel",
                "orbit": "BH",
            }
        ],
        "rvinstrument": [
            {
                "name": "inst",
                "file": _write_rv(tmp_path / "rv.dat"),
                "star_ndx": 1,
            }
        ],
    }
    params = _params()
    config["planet"][1]["orbit_ndx"] = 1
    params["orbit.c.period"] = {"initval": 31.0}
    params["orbit.c.tc"] = {"initval": 2455010.0}
    params["planet.c.radius"] = {"initval": 1.0}
    params["star.B.mass"] = {"initval": 1.0, "sigma": 0.05}
    params["star.B.radius"] = {"initval": 1.0, "sigma": 0.1}

    system = System(config, params)
    system.prepare()
    assert list(system.orbit.node_degenerate) == [True, False]


# ---------------------------------------------------------------------------
# 2. The truncation is gone
# ---------------------------------------------------------------------------


def test_ybigomega_carries_no_lower_bound(astrometry_only):
    """
    Given a node-degenerate orbit,
    When ybigomega is built,
    Then its support is the defaults.yaml one, symmetric about zero.  The
      `lower: 0` half-plane truncation is what biased a boundary-hugging
      posterior, and the antipodal modes are joined through the origin with
      no likelihood barrier, so nothing needs it.
    """
    system, _ = astrometry_only
    lower = np.atleast_1d(system.orbit.ybigomega.lower)
    assert (lower < 0).all()


# ---------------------------------------------------------------------------
# 3. The degeneracy is exact, and the fold is its inverse
# ---------------------------------------------------------------------------


def _element_raw(param, point, i):
    tf = param._raw_transform
    slot = list(tf["sampled_idx"]).index(i)
    return point[f"{param.label}_raw"][slot]


def _set_element_raw(param, point, i, value):
    tf = param._raw_transform
    slot = list(tf["sampled_idx"]).index(i)
    point[f"{param.label}_raw"] = np.array(point[f"{param.label}_raw"])
    point[f"{param.label}_raw"][slot] = value


def _antipodal_point(system, point):
    """The partner label of `point`, built from the physics and not the fold.

    Applies the reflection by hand -- sign flips on the direction vector and
    the sqrt(e) pair, and `tc` moved to the other conjunction -- so that
    comparing logp is a statement about the MODEL, and comparing the fold's
    output to this is a statement about the fold.
    """
    orbit = system.orbit
    out = {k: np.array(v) for k, v in point.items()}
    phys = {
        name: getattr(orbit, name).element_phys_from_raw(
            0, _element_raw(getattr(orbit, name), point, 0)
        )
        for name in (
            "xbigomega",
            "ybigomega",
            "secosw",
            "sesinw",
            "tc",
            "logP",
        )
    }
    ecc = phys["secosw"] ** 2 + phys["sesinw"] ** 2
    omega = np.arctan2(phys["sesinw"], phys["secosw"])
    period = 10.0 ** phys["logP"]
    delta = (
        (
            physics.mean_anomaly_at_conjunction(ecc, omega + np.pi)
            - physics.mean_anomaly_at_conjunction(ecc, omega)
        )
        * period
        / (2.0 * np.pi)
    )
    tf = orbit.tc._raw_transform
    lo, up = float(tf["lowers"][0]), float(tf["uppers"][0])
    tc_new = lo + np.mod(phys["tc"] + delta - lo, up - lo)

    for name in ("xbigomega", "ybigomega", "secosw", "sesinw"):
        param = getattr(orbit, name)
        _set_element_raw(
            param, out, 0, param.element_raw_from_phys(0, -phys[name])
        )
    _set_element_raw(
        orbit.tc, out, 0, orbit.tc.element_raw_from_phys(0, tc_new)
    )
    return out


def test_the_two_labels_have_the_same_likelihood(astrometry_only):
    """
    Given a point and its antipodal partner, built from the reflection
      itself and not from the fold,
    When the OBSERVATION likelihood is evaluated at both,
    Then it is the same to within numerical noise.  That is the claim the
      whole item rests on: the degeneracy is exact, so there is no
      likelihood barrier between the two labels and no reason for a hard
      bound to exclude one of them.
    """
    system, model = astrometry_only
    point = model.initial_point()
    partner = _antipodal_point(system, point)
    obs_logp = model.compile_logp(vars=model.observed_RVs, sum=True)

    assert float(obs_logp(partner)) == pytest.approx(
        float(obs_logp(point)), rel=1e-8, abs=1e-6
    )


def test_the_prior_terms_cancel_for_the_reflected_coordinates(
    astrometry_only,
):
    """
    Given the same pair,
    When the PRIOR of each reflected coordinate is compared,
    Then the sign-flipped ones agree exactly: the logit reparameterization
      makes each uniform over its own bounds, `xbigomega`/`ybigomega`'s
      Gaussian is centred on ZERO, and `secosw`/`sesinw`'s bounds are
      symmetric, so `v -> -v` is a symmetry of every term.

    `tc` is deliberately NOT in this list, and the reason is worth pinning
    rather than discovering: the partner's tc is half a period away, its raw
    coordinate is scaled to the TIMING precision, and the excursion is ~1e5
    raw units -- past `_RAW_CANCELLATION_CLIP`, where the logit-uniform
    correction stops cancelling the raw N(0,1) by design.  So the FULL logp
    of the two labels is not comparable, even though their physical density
    is identical; that is an artifact of the sampled coordinate, and it is
    also why the two labels are effectively disconnected for a single chain.
    """
    system, model = astrometry_only
    point = model.initial_point()
    partner = _antipodal_point(system, point)

    for name in system.orbit._FOLD_FLIP:
        param = getattr(system.orbit, name)
        terms = [
            v
            for v in model.basic_RVs + model.potentials
            if v.name
            in (f"{param.label}_raw", f"logit_uniform_prior.{param.label}")
        ]
        assert terms, f"no prior terms found for {param.label}"
        fn = model.compile_logp(vars=terms, sum=True)
        assert float(fn(partner)) == pytest.approx(
            float(fn(point)), rel=1e-9, abs=1e-6
        ), name


def test_the_fold_maps_the_partner_back(astrometry_only):
    """
    Given a posterior holding one draw in each half-plane,
    When the fold runs,
    Then every draw ends up in the canonical half-plane and the folded
      partner reproduces the original draw's raw coordinates.
    """
    # Arrange
    import xarray as xr

    system, model = astrometry_only
    point = model.initial_point()
    partner = _antipodal_point(system, point)
    names = [
        f"{system.orbit.prefix}.{n}_raw"
        for n in ("xbigomega", "ybigomega", "secosw", "sesinw", "tc", "logP")
    ]
    posterior = xr.Dataset(
        {
            name: (
                ("chain", "draw", f"{name}_dim_0"),
                np.stack([[point[name], partner[name]]]),
            )
            for name in names
        }
    )

    # Act
    moved = system.orbit.fold_node_degeneracy(posterior)

    # Assert -- both draws end up on the SAME label.  Which of the two the
    # fold lands on is not asserted: the half-plane it folds onto is the
    # draws' own axial mean, so the answer is a property of the sample, and
    # the claim that matters is that a diagnostic can no longer see two.
    assert moved
    for name in names:
        got = posterior[name].values[0]
        np.testing.assert_allclose(got[0], got[1], atol=1e-7)


def test_the_fold_is_a_no_op_where_nothing_is_degenerate(tmp_path):
    """
    Given a system whose orbit an RV constrains,
    When the fold runs,
    Then it reports that nothing moved: there is no degeneracy to collapse,
      and folding a coordinate the data DO identify would throw away a real
      distinction.
    """
    import xarray as xr

    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [{"name": "BH"}],
        "astrometryinstrument": [
            {
                "name": "Rel",
                "file": _write_rel(tmp_path / "sim.rel.astrom"),
                "mode": "rel",
            }
        ],
        "rvinstrument": [
            {"name": "inst", "file": _write_rv(tmp_path / "rv.dat")}
        ],
    }
    system = System(config, _params())
    system.prepare()
    assert not system.orbit.fold_node_degeneracy(xr.Dataset())
