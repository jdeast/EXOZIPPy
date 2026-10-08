"""The `fitmsini` mass coordinate and the retirement of `mass_parameterization`.

Review 2.14.9: an RV-only orbit measures m sin i and nothing measures the
inclination, so sampling (mass, cos i) is sampling a banana whose tail
(P(M > X) ~ (2/pi) msini/X) the sampler under-explores.  `fitmsini` samples
(msini, cos i) instead, derives mass = msini / sin i, and adds the -log sin i
Jacobian so the prior stays uniform in (mass, cos i) -- the same posterior,
in the coordinates the data measure.  It is the DEFAULT for a planet whose
mass RVs measure and whose orbit's inclination nothing does.

The mass coordinate is an n-way choice spelled as exclusive booleans
(`fitlogq`, `fitmsini`; neither = the signed linear mass), and the enum
`mass_parameterization:` it replaced is refused with its migration.

See src/exozippy/components/star/star.md, "Planet mass parametrization".
"""

import os

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import yaml

from exozippy.components.orbit.orbit import inclination_constrained_orbits
from exozippy.components.planet import physics
from exozippy.components.planet.planet import (
    Planet,
    lens_companion_samples_log_q,
    parse_mass_flags,
)
from exozippy.system import System

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


# ---------------------------------------------------------------------------
# 1. The per-planet decision (no model build)
# ---------------------------------------------------------------------------
class _FakeOrbit:
    prefix = "orbit"

    def __init__(
        self, n=1, inc_modes=None, keplerian=(), xallarap=(), pinned=None
    ):
        self._pinned = pinned or {}
        self.n_elements = n
        self.names = [f"o{i}" for i in range(n)]
        self._inc_modes = inc_modes or ["cosi"] * n
        self._kep = set(keplerian)
        self._xal = set(xallarap)

    def star_membership(self, star_idx):
        return [(i, "primary") for i in range(self.n_elements)]

    def bodies(self, i):
        return [("star", 0), ("planet", int(i))]

    def inclination_modes(self, system):
        return list(self._inc_modes)

    def _lens_keplerian_orbits(self, system):
        return set(self._kep)

    def _lens_xallarap_orbits(self, system):
        return set(self._xal)

    def _user_pinned(self, index, params):
        return list(self._pinned.get(index, ()))


class _FakeRV:
    star_ndx = [0]


class _FakeAstrometry:
    def __init__(self, mode):
        self.modes = [mode]
        self.rel_orbit = [0 if mode == "rel" else None]
        self.config = [{"star_ndx": 0}]


class _FakeSystem:
    def __init__(self, config=None, **comps):
        self.active_components = comps
        self.config = config or {}


class _FakeConfigManager:
    def __init__(self, user_params=None):
        self.user_params = user_params if user_params is not None else {}


def _modes(planet_cfg, user_params=None, config=None, **comps):
    comp = Planet(planet_cfg, config_manager=_FakeConfigManager(user_params))
    comp.build_maps()
    comp._resolve_mass_parameterization(_FakeSystem(config, **comps))
    return comp.mass_parameterizations


def _values(model, nodes):
    """`nodes` rewritten onto the model's VALUE variables.  A Parameter's
    `.value` hangs off the free RVs, so compiling it directly against
    `value_vars` evaluates a prior DRAW, not the point passed in."""
    return model.replace_rvs_by_values(list(nodes))


def _rv_only(**extra):
    return dict(rvinstrument=_FakeRV(), orbit=_FakeOrbit(), **extra)


def test_an_rv_only_planet_defaults_to_msini():
    """
    Given RVs measuring the planet's orbit and nothing measuring its
      inclination,
    When the mass coordinate is resolved,
    Then it samples (m sin i, cos i).
    """
    assert _modes([{"name": "b"}], **_rv_only()) == ["msini"]


@pytest.mark.parametrize(
    "extra, config",
    [
        ({"transit": object()}, None),
        ({"astrometryinstrument": _FakeAstrometry("gaia")}, None),
        ({"astrometryinstrument": _FakeAstrometry("abs")}, None),
        ({"astrometryinstrument": _FakeAstrometry("rel")}, None),
        ({}, {"rvinstrument": [{"rm": "o0"}]}),
        ({}, {"dopptom": [{"orbit": "o0"}]}),
    ],
    ids=["transit", "gaia", "abs", "rel", "rm", "dopptom"],
)
def test_an_inclination_measurement_keeps_the_linear_mass(extra, config):
    """
    Given RVs AND a dataset that measures the orbit's inclination,
    When the mass coordinate is resolved,
    Then the signed linear mass is kept: the mass is measured, not m sin i.
    """
    assert _modes([{"name": "b"}], config=config, **_rv_only(**extra)) == [
        "linear"
    ]


@pytest.mark.parametrize("which", ["keplerian", "xallarap"])
def test_microlensing_orbital_motion_measures_the_inclination(which):
    """A keplerian lens or source orbit consumes the sky geometry per epoch,
    so it constrains cos i just as astrometry does."""
    orbit = _FakeOrbit(**{which: [0]})
    system = _FakeSystem(rvinstrument=_FakeRV(), orbit=orbit)
    assert inclination_constrained_orbits(system, orbit) == {0}
    assert _modes([{"name": "b"}], rvinstrument=_FakeRV(), orbit=orbit) == [
        "linear"
    ]


def test_a_pinned_inclination_keeps_the_linear_mass():
    """`sigma: 0` on cos i states the inclination outright, so m sin i
    coordinates buy nothing."""
    orbit = _FakeOrbit(pinned={0: ["cosi"]})
    system = _FakeSystem(rvinstrument=_FakeRV(), orbit=orbit)
    assert inclination_constrained_orbits(system, orbit) == {0}
    assert _modes([{"name": "b"}], rvinstrument=_FakeRV(), orbit=orbit) == [
        "linear"
    ]


def test_inclination_predicate_is_empty_for_rvs_alone():
    orbit = _FakeOrbit(n=2)
    system = _FakeSystem(rvinstrument=_FakeRV(), orbit=orbit)
    assert inclination_constrained_orbits(system, orbit) == set()


def test_a_pinned_mass_keeps_the_linear_default():
    """
    Given a user `planet.b.mass: {sigma: 0}`,
    When the default is resolved,
    Then the planet keeps the linear mass -- in msini coordinates the mass
      is derived and the pin would be dropped (the orbit's
      _pin_blocks_default rule).
    """
    up = {"planet.0.mass": {"initval": 1.0, "sigma": 0.0}}
    assert _modes([{"name": "b"}], user_params=up, **_rv_only()) == ["linear"]


def test_explicit_flags_win_over_the_default():
    assert _modes([{"name": "b", "fitmsini": False}], **_rv_only()) == [
        "linear"
    ]
    assert _modes([{"name": "b", "fitlogq": True}], **_rv_only()) == ["log_q"]
    # fitlogq: false rules out log_q only; the msini default stands.
    assert _modes([{"name": "b", "fitlogq": False}], **_rv_only()) == ["msini"]
    # An explicit fitmsini is honored where the default would not choose it.
    assert _modes(
        [{"name": "b", "fitmsini": True}],
        **_rv_only(transit=object()),
    ) == ["msini"]


def test_both_flags_true_raises_naming_both():
    with pytest.raises(ValueError, match="fitlogq and fitmsini are both"):
        parse_mass_flags({"fitlogq": True, "fitmsini": True}, "planet.b")


@pytest.mark.parametrize(
    "value, translation",
    [
        ("log_q", "'fitlogq: true'"),
        ("linear", "'fitlogq: false' and 'fitmsini: false'"),
    ],
)
def test_the_retired_enum_raises_with_its_migration(value, translation):
    """`mass_parameterization:` is refused at the config boundary (the
    planet's construction), with the spelling that replaces it."""
    with pytest.raises(ValueError) as exc:
        Planet(
            [{"name": "b", "mass_parameterization": value}],
            config_manager=_FakeConfigManager(),
        )
    msg = str(exc.value)
    assert "mass_parameterization" in msg and "replaced" in msg
    assert translation in msg


def test_fitmsini_on_a_chord_orbit_raises():
    """The chord derives cos i from a/R*, which depends on the mass, while
    msini derives the mass from cos i: a dependency cycle, refused at
    stage 3 rather than left to the build graph."""
    orbit = _FakeOrbit(inc_modes=["chord"])
    with pytest.raises(ValueError, match="dependency cycle"):
        _modes(
            [{"name": "b", "fitmsini": True}],
            rvinstrument=_FakeRV(),
            orbit=orbit,
        )
    # ...while the DEFAULT just steps down to the linear mass.
    assert _modes([{"name": "b"}], rvinstrument=_FakeRV(), orbit=orbit) == [
        "linear"
    ]


def test_a_chord_on_ANOTHER_orbit_also_rules_msini_out():
    """The build orders parameter VECTORS: orbit.cosi depends on planet.mass
    through the chord orbit's element, and an msini planet's mass reads
    orbit.cosi, so the cycle crosses orbits (it recursed in add_parameter
    for two RV orbits with one fitvcve -> fitchord)."""
    orbit = _FakeOrbit(n=2, inc_modes=["chord", "cosi"])
    planets = [{"name": "b"}, {"name": "c", "orbit_ndx": 1}]
    assert _modes(planets, rvinstrument=_FakeRV(), orbit=orbit) == [
        "linear",
        "linear",
    ]
    planets[1]["fitmsini"] = True
    with pytest.raises(ValueError, match="dependency cycle"):
        _modes(planets, rvinstrument=_FakeRV(), orbit=orbit)


def test_fitmsini_without_an_orbit_raises():
    with pytest.raises(ValueError, match="needs an orbit"):
        _modes([{"name": "b", "fitmsini": True}])


def test_lens_companion_log_q_reader():
    """The star/mulensevent readers ask before the planet resolves; a lens
    body defaults to log_q unless the entry turns it off or asks for msini."""
    assert lens_companion_samples_log_q({}, "planet.b")
    assert lens_companion_samples_log_q({"fitlogq": True}, "planet.b")
    assert not lens_companion_samples_log_q({"fitlogq": False}, "planet.b")
    assert not lens_companion_samples_log_q({"fitmsini": True}, "planet.b")


# ---------------------------------------------------------------------------
# 2. The built model (no data: everything but msini/mass and cos i pinned)
# ---------------------------------------------------------------------------
def _build(planet_extra, extra_params=None, n_planets=1):
    names = ["b", "c"][:n_planets]
    planets = [{"name": nm, "orbit_ndx": i} for i, nm in enumerate(names)]
    for p, extra in zip(planets, planet_extra):
        p.update(extra)
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": planets,
        "orbit": [{"name": nm} for nm in names],
    }
    params = {
        "star.A.logmass": {"initval": 0.0, "sigma": 0.0},
        "star.A.radius": {"initval": 1.0, "sigma": 0.0},
    }
    for i, nm in enumerate(names):
        params.update(
            {
                f"orbit.{nm}.logP": {"initval": 0.5 + i, "sigma": 0.0},
                f"orbit.{nm}.tc": {"initval": 2455000.0, "sigma": 0.0},
                f"orbit.{nm}.secosw": {"initval": 0.0, "sigma": 0.0},
                f"orbit.{nm}.sesinw": {"initval": 0.0, "sigma": 0.0},
                f"planet.{nm}.radius": {"initval": 1.0, "sigma": 0.0},
                f"orbit.{nm}.cosi": {"initval": 0.3},
                f"planet.{nm}.mass": {"initval": 2.0},
            }
        )
    params.update(extra_params or {})
    system = System(config, params)
    system.prepare()
    return system, system.build_model()


@pytest.fixture(scope="module")
def msini_model():
    return _build([{"fitmsini": True}])


@pytest.fixture(scope="module")
def linear_model():
    return _build([{"fitlogq": False, "fitmsini": False}])


def test_msini_is_sampled_and_mass_derived(msini_model):
    system, model = msini_model
    planet = system.planet
    assert planet.mass_parameterizations == ["msini"]
    free = {v.name for v in model.free_RVs}
    assert free == {"planet.msini_raw", "orbit.cosi_raw"}
    assert planet.mass.is_derived.tolist() == [True]
    assert planet.msini.is_sampled.tolist() == [True]
    # mass is a node in the trace, as when it is sampled.
    assert "planet.mass" in {d.name for d in model.deterministics}
    # msini takes planet.mass's HARD bounds (the overrides channel).
    lo, hi = (
        float(np.atleast_1d(x)[0])
        for x in (planet.msini.lower, planet.msini.upper)
    )
    assert lo == pytest.approx(planet.mass.to_internal(-1000.0, index=0))
    assert hi == pytest.approx(planet.mass.to_internal(260000.0, index=0))
    assert "planet.msini_jacobian" in {p.name for p in model.potentials}


def test_a_mass_seed_back_solves_into_msini(msini_model):
    """Eq(msini, mass * sini) at rank 5: the user's mass seed reaches the
    sampled coordinate (the ledger start of msini is mass x sin i), while the
    mass keeps the value the user wrote."""
    system, model = msini_model
    sini = np.sqrt(1.0 - 0.3**2)
    mass0 = system.planet.mass.to_internal(2.0, index=0)
    assert float(np.atleast_1d(system.planet.mass.initval)[0]) == (
        pytest.approx(mass0, rel=1e-12)
    )
    assert float(np.atleast_1d(system.planet.msini.initval)[0]) == (
        pytest.approx(mass0 * sini, rel=1e-9)
    )
    # ...and the derived node is msini / sin i at the start point.
    f = pytensor.function(
        model.value_vars,
        _values(
            model,
            [
                system.planet.mass.value,
                system.planet.msini.value,
                system.orbit.cosi.value,
            ],
        ),
        on_unused_input="ignore",
    )
    start = model.initial_point()
    mass, msini, cosi = (
        float(np.atleast_1d(x)[0])
        for x in f(*[start[v.name] for v in model.value_vars])
    )
    assert mass == pytest.approx(msini / np.sqrt(1.0 - cosi**2), rel=1e-12)


def test_no_msini_bounds_leak_onto_reported_msini(linear_model):
    """msini is REPORTED on a linear planet: no bound, so no soft barrier
    (the overrides carry NaN there and defaults.yaml has none)."""
    system, model = linear_model
    assert system.planet.mass_parameterizations == ["linear"]
    assert system.planet.msini.lower is None
    assert system.planet.msini.upper is None
    assert "planet.msini_jacobian" not in {p.name for p in model.potentials}
    assert not any("planet.msini" in p.name for p in model.potentials)


def _physical(system, model, raw_name):
    """(logp, mass, cosi, |d(mass, cosi)/d(raw, cosi_raw)|) as functions of
    the two free raw coordinates."""
    vv = {v.name: v for v in model.value_vars}
    raw, craw = vv[raw_name], vv["orbit.cosi_raw"]
    mass, cosi = _values(
        model, [system.planet.mass.value[0], system.orbit.cosi.value[0]]
    )

    def d(y, x):
        return pytensor.grad(y, x, disconnected_inputs="ignore")[0]

    det = d(mass, raw) * d(cosi, craw) - d(mass, craw) * d(cosi, raw)
    logdet = pt.log(pt.abs(det))
    jterm = [p for p in model.potentials if p.name == "planet.msini_jacobian"]
    outs = [model.logp(), mass, cosi, logdet]
    outs.append(_values(model, [jterm[0]])[0] if jterm else pt.constant(0.0))
    return pytensor.function([raw, craw], outs, on_unused_input="ignore")


@pytest.mark.parametrize("which", ["msini", "linear"])
def test_the_implied_prior_on_mass_and_cosi_is_flat(
    which, msini_model, linear_model
):
    """
    Given no likelihood, everything but the mass coordinate and cos i pinned,
    When the model's density is carried from its sampled raw coordinates to
      (mass, cos i) -- log p(mass, cos i) = logp(raw) - log|det J|,
    Then it is constant over the interior: the prior is uniform in mass and
      cos i in BOTH coordinates, so the switch is answer-preserving.  And
      without the Jacobian term the msini density is NOT flat -- it varies as
      log sin i, which is what the term is for (the sign is checked here as a
      property, not argued).
    """
    system, model = msini_model if which == "msini" else linear_model
    raw_name = f"planet.{which if which == 'msini' else 'mass'}_raw"
    f = _physical(system, model, raw_name)
    dens, nojac, cosis = [], [], []
    for r in np.linspace(-0.004, 0.004, 5):
        for c in np.linspace(-40.0, 40.0, 9):
            lp, mass, cosi, logdet, jterm = f(np.array([r]), np.array([c]))
            assert 0.0 < mass < 0.05  # interior: soft barriers are inert
            dens.append(lp - logdet)
            nojac.append(lp - jterm - logdet)
            cosis.append(cosi)
    dens = np.array(dens)
    assert np.ptp(dens) < 1e-6, np.ptp(dens)
    if which == "msini":
        nojac = np.array(nojac)
        assert np.ptp(nojac) > 0.05
        # ...and what is left is exactly +log sin i.
        resid = nojac - 0.5 * np.log(1.0 - np.array(cosis) ** 2)
        assert np.ptp(resid) < 1e-6


def test_the_jacobian_matches_a_finite_difference(msini_model):
    """
    Given the built msini model,
    When d mass / d msini is taken by central finite differences at fixed
      cos i,
    Then it equals 1/sin i, and the msini_jacobian potential is its log --
      the change of variables the term claims, measured on the model's own
      mass node.  The model's dlogp agrees with finite differences of its
      logp too, so the floored radicand leaves the gradient exact inside.
    """
    system, model = msini_model
    vv = {v.name: v for v in model.value_vars}
    jterm = [p for p in model.potentials if p.name == "planet.msini_jacobian"][
        0
    ]
    outs = _values(
        model,
        [
            system.planet.msini.value[0],
            system.planet.mass.value[0],
            system.orbit.cosi.value[0],
            jterm,
        ],
    )
    f = pytensor.function(list(vv.values()), outs, on_unused_input="ignore")
    names = list(vv)
    for c in (-1.2, 0.0, 0.7):
        pt0 = {
            "planet.msini_raw": np.array([1e-3]),
            "orbit.cosi_raw": np.array([c]),
        }
        h = 1e-6

        def ev(dr):
            p = dict(pt0)
            p["planet.msini_raw"] = pt0["planet.msini_raw"] + dr
            return f(*[p[n] for n in names])

        s_p, m_p, ci, _ = ev(h)
        s_m, m_m, _, _ = ev(-h)
        _, _, _, jt = ev(0.0)
        dm_ds = (m_p - m_m) / (s_p - s_m)
        sini = np.sqrt(1.0 - ci**2)
        assert dm_ds == pytest.approx(1.0 / sini, rel=1e-6)
        assert jt == pytest.approx(np.log(dm_ds), rel=1e-6, abs=1e-9)

    logp = model.compile_logp()
    dlogp = model.compile_dlogp()
    point = {
        "planet.msini_raw": np.array([2e-3]),
        "orbit.cosi_raw": np.array([0.4]),
    }
    grad = np.concatenate([np.atleast_1d(g) for g in [dlogp(point)]]).ravel()
    order = [v.name for v in model.value_vars]
    for k, name in enumerate(order):
        e = 1e-6
        hi = dict(point)
        lo = dict(point)
        hi[name] = point[name] + e
        lo[name] = point[name] - e
        fd = (logp(hi) - logp(lo)) / (2 * e)
        assert grad[k] == pytest.approx(fd, rel=1e-4, abs=1e-6), name


def test_the_floor_keeps_the_gradient_finite_at_an_edge_on_cos_i():
    """|cos i| -> 1 rounds 1 - cos^2 i to 0.0 in float64; the strictly
    positive radicand floor keeps the mass, the Jacobian and their gradients
    finite there (the CLAUDE.md where-trap rule)."""
    c = pt.dscalar("c")
    s = pt.dscalar("s")
    m = physics.calc_mass_from_msini(s, c)
    j = physics.msini_log_jacobian(c)
    f = pytensor.function(
        [s, c], [m, j, pt.grad(m, c), pt.grad(j, c), pt.grad(m, s)]
    )
    for edge in (1.0, -1.0, 1.0 - 1e-17):
        assert all(np.isfinite(v) for v in f(1e-3, edge))


def test_the_prior_column_says_what_the_jacobian_did(msini_model):
    system, _ = msini_model
    txt = system.planet.msini.get_prior_str(0, latex=False)
    assert "1/sin i" in txt


def test_a_mixed_system_reports_both_ways(tmp_path):
    """
    Given one planet sampling msini and one sampling its linear mass,
    When the model is built,
    Then there is no per-parameter cycle (msini is REPORTED on the linear
      planet), and every reported value obeys msini = mass * sin i.
    """
    system, model = _build(
        [{"fitmsini": True}, {"fitlogq": False, "fitmsini": False}],
        n_planets=2,
    )
    assert system.planet.mass_parameterizations == ["msini", "linear"]
    assert system.planet.msini.is_sampled.tolist() == [True, False]
    assert system.planet.mass.is_sampled.tolist() == [False, True]
    start = model.initial_point()
    f = pytensor.function(
        model.value_vars,
        _values(
            model,
            [
                system.planet.msini.value,
                system.planet.mass.value,
                system.orbit.cosi.value,
            ],
        ),
        on_unused_input="ignore",
    )
    # The REPORTED element is patched in a deferred pass; read the patched
    # node off the model's deterministics.
    msini_det = [d for d in model.deterministics if d.name == "planet.msini"][
        0
    ]
    g = pytensor.function(
        model.value_vars,
        _values(model, [msini_det])[0],
        on_unused_input="ignore",
    )
    args = [start[v.name] for v in model.value_vars]
    _, mass, cosi = (np.atleast_1d(x) for x in f(*args))
    msini = np.atleast_1d(g(*args))
    np.testing.assert_allclose(msini, mass * np.sqrt(1 - cosi**2), rtol=1e-12)
    assert np.isfinite(model.compile_logp()(start))


def _example(example, cfg, monkeypatch, edit=None):
    monkeypatch.chdir(os.path.join(EXAMPLES, example))
    with open(cfg) as fh:
        config = yaml.safe_load(fh)
    with open(config["parameter_file"]) as fh:
        params = yaml.safe_load(fh) or {}
    config["parameter_file"] = None
    if edit is not None:
        edit(config)
    system = System(config, params)
    system.prepare()
    return system


def test_fitmsini_and_fitvcve_are_independent(monkeypatch):
    """V_c/V_e is an eccentricity coordinate and msini a mass one: an RV-only
    orbit with an explicit fitvcve still samples msini, and builds."""

    def edit(config):
        config["orbit"][0]["fitvcve"] = True
        config["orbit"][0]["fitchord"] = False

    system = _example("kelt4", "kelt4_rvonly.yaml", monkeypatch, edit)
    assert system.planet.mass_parameterizations == ["msini"]
    assert system.orbit.ecc_modes == ["vcve"]
    model = system.build_model()
    assert np.isfinite(model.compile_logp()(model.initial_point()))


# ---------------------------------------------------------------------------
# 3. The shipped RV-only examples flip; the rest keep their coordinates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "example, cfg, mode",
    [
        ("kelt4", "kelt4_rvonly.yaml", "msini"),
        ("hd80606", "hd80606_rvonly.yaml", "msini"),
        # The same RVs plus the TESS transit, which measures the inclination.
        ("hd80606", "hd80606.yaml", "linear"),
    ],
)
def test_the_shipped_rv_only_examples(example, cfg, mode, monkeypatch):
    def edit(config):
        # The mass coordinate is decided by what measures the orbit, never by
        # the star's evolutionary model, and the ~128 MB MIST grid is not
        # shipped (CI never downloads it), so the hd80606 configs drop it.
        config.pop("evolutionarymodel", None)

    system = _example(example, cfg, monkeypatch, edit)
    assert system.planet.mass_parameterizations == [mode]
