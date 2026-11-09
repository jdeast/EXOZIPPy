"""The red-clump reference (components/redclump/redclump.md), the SED
flux-source seam it is read through, the galactic model's sight-line
queries, and the one sight-line R_V (sed.md).

Real NextGen tables (the conftest prewarm fetches them); the fitted-R_V
cases use test_sed_fit_rv_dust's blackbody runtime grid, so the numbers
they pin are about the machinery.
"""

import numpy as np
import pytest
import yaml

from exozippy.components.component import SEDFluxSource
from exozippy.components.galacticmodel import koshimoto21 as k21
from exozippy.components.star.star import Star
from exozippy.system import System

# Baade's window
RA, DEC = 270.9, -30.0

# The bar on six bulge sight lines (dust_stack design F3): mean / sd / mode
# of the distance modulus.  genulens's own coded clump formula is within
# 0.08 mag of every mean.
BAR = {
    (1.0, -3.9): (14.5277, 0.2383, 14.4363),
    (0.0, -3.5): (14.5782, 0.2278, 14.5583),
    (5.0, -3.0): (14.4211, 0.2375, 14.4173),
    (-5.0, -3.0): (14.7534, 0.2415, 14.7676),
    (1.0, -1.5): (14.5430, 0.1695, 14.5001),
    (0.5, -6.0): (14.5087, 0.2989, 14.3409),
}


@pytest.mark.parametrize("lb", sorted(BAR))
def test_bar_distance_moments_reproduce_the_ports_bar_and_genulens(lb):
    """
    Given six bulge sight lines,
    When the bar's distance-modulus moments are computed from the port,
    Then they are the recorded values, and the mean is within 0.15 mag of
      genulens's own red-clump distance formula.
    """
    got = k21.bar_distance_moments(*lb)
    np.testing.assert_allclose(got, BAR[lb], atol=1e-4)
    assert abs(got[0] - k21.red_clump_distance_modulus(*lb)) < 0.15


# ---------------------------------------------------------------------------
# A clump system
# ---------------------------------------------------------------------------
V, I = "Generic/Johnson.V", "Generic/Cousins.I"


def _rows():
    return [
        {
            "name": I,
            "mag": 15.80,
            "err": 0.05,
            "photType": {"pos": ["RedClump"]},
        },
        {
            "name": V,
            "mag": 2.60,
            "err": 0.03,
            "neg_filter": I,
            "photType": {"pos": ["RedClump"], "neg": ["RedClump"]},
        },
        {
            "name": I,
            "mag": 2.0,
            "err": 0.02,
            "photType": {"pos": ["Source"], "neg": ["RedClump"]},
        },
        {
            "name": V,
            "mag": -0.3,
            "err": 0.01,
            "color_filter": I,
            "photType": {"pos": ["Source"], "neg": ["RedClump"]},
        },
    ]


def _config(tmp_path, rows=None, clump=None, sed=None, stars=("Source",)):
    path = tmp_path / "rc.sed"
    path.write_text(
        yaml.safe_dump(
            {"model": "NextGen", "filters": _rows() if rows is None else rows}
        )
    )
    return {
        "star": [{"name": s, "mist": False} for s in stars],
        "sed": {"file": str(path), **(sed or {})},
        "galacticmodel": [{"anchor_idx": 0}],
        "redclump": [
            {"name": "RedClump", "bands": ["I"], "colors": [["V", "I"]]}
            if clump is None
            else clump
        ],
        "band": [
            {"name": "I", "filter": I, "star_constrains_ld": False},
            {"name": "V", "filter": V, "star_constrains_ld": False},
        ],
    }


def _params(ra=RA, dec=DEC, **extra):
    return {
        "star.Source.ra": {"initval": ra, "sigma": 0},
        "star.Source.dec": {"initval": dec, "sigma": 0},
        **extra,
    }


def _prepared(tmp_path, params=None, **kw):
    system = System(_config(tmp_path, **kw), params or _params())
    system.prepare()
    return system


@pytest.fixture(scope="module")
def clump_model(tmp_path_factory):
    system = _prepared(tmp_path_factory.mktemp("rc"))
    return system, system.build_model()


def test_the_clump_is_one_more_sed_source(clump_model):
    """
    Given a star and a clump the SED rows name,
    When the model is built,
    Then the source axis is [stars..., clump], the clump is read by the
      rows but is no SEDReader star, its prediction comes out of the one
      BC evaluation, and the start logp and gradient are finite.
    """
    system, model = clump_model
    sed = system.sed
    assert sed.source_names == ["Source", "RedClump"]
    assert sed.pos_matrix.shape == (4, 2)
    assert sed.provider_readers(system) == {"RedClump": ["sed photometry"]}
    assert {r.star for r in sed.photometric_readers(system)} == {0}
    ip = model.initial_point()
    assert np.isfinite(model.compile_logp()(ip))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))
    names = {v.name for v in model.free_RVs}
    for p in ("teff", "feh", "mi", "distance", "av"):
        assert f"redclump.{p}_raw" in names, p
    assert "redclump.logg_raw" not in names  # pinned


def test_vi0_and_radius_are_the_tables_at_av_zero(clump_model):
    """
    Given the clump at its start,
    When vi0 and radius are evaluated,
    Then vi0 is BC_I - BC_V at A_V = 0 off the SED's grid, and the radius
      returns M_I = mi through the same BC to 1e-10 mag.
    """
    system, model = clump_model
    rc, sed = system.redclump, system.sed
    ip = model.initial_point()
    t, g, z, mi = (
        float(np.atleast_1d(x)[0])
        for x in _at(
            model,
            [rc.teff.value, rc.logg.value, rc.feh.value, rc.mi.value],
            ip,
        )
    )
    vi0, radius = (
        float(np.atleast_1d(x)[0])
        for x in _at(
            model, [rc.vi0.value, rc.radius.value], model.initial_point()
        )
    )
    bc_v, bc_i = sed.bc_np(V, t, g, z, 0.0), sed.bc_np(I, t, g, z, 0.0)
    assert vi0 == pytest.approx(bc_i - bc_v, abs=1e-9)
    from exozippy.constants import LOG_L0_CONST, LUM_CONST

    mbol = -2.5 * np.log10(LUM_CONST * radius**2 * t**4) + LOG_L0_CONST
    assert mbol - bc_i == pytest.approx(mi, abs=1e-10)


def test_the_distance_prior_is_the_bar_and_av_starts_from_the_clump_color(
    clump_model,
):
    system, _ = clump_model
    rc = system.redclump
    mean, _sd, _mode = rc.bar_moments
    assert rc.distance.mu == pytest.approx(10 ** (0.2 * mean + 1))
    assert rc.distance.sigma == pytest.approx(
        0.1 * np.log(10) / 5 * 10 ** (0.2 * mean + 1)
    )
    sed = system.sed
    t, g, z, av = (
        float(np.atleast_1d(getattr(rc, p).initval)[0])
        for p in ("teff", "logg", "feh", "av")
    )
    # the (V-I)_RC row is reproduced at the av start
    assert sed.bc_np(I, t, g, z, av) - sed.bc_np(V, t, g, z, av) == (
        pytest.approx(2.60, abs=1e-4)
    )


# ---------------------------------------------------------------------------
# The raises
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c.pop("galacticmodel"), "galacticmodel"),
        (lambda c: c["redclump"].append({"name": "Two"}), "exactly one"),
        (lambda c: c["redclump"][0].update(color="x"), "unknown key"),
        (
            lambda c: c["redclump"][0].update(clump_constrains_source_av="no"),
            "YAML boolean",
        ),
        (lambda c: c["redclump"][0].update(bands=["K"]), "no band: block"),
        (
            lambda c: c["redclump"][0].update(colors=[["V", "V"]]),
            "two different",
        ),
    ],
)
def test_construction_raises(tmp_path, mutate, match):
    cfg = _config(tmp_path)
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        System(cfg, _params())


def test_a_clump_off_the_bulge_raises(tmp_path):
    with pytest.raises(ValueError, match="no red clump"):
        _prepared(tmp_path, params=_params(ra=120.0, dec=40.0))


def test_a_clump_nothing_reads_raises(tmp_path):
    rows = [
        {"name": I, "mag": 15.0, "err": 0.05, "photType": {"pos": ["Source"]}}
    ]
    with pytest.raises(ValueError, match="read by no SED row and no prior"):
        _prepared(tmp_path, rows=rows, clump={"name": "RedClump"})


def test_a_prior_on_an_extinction_element_reads_the_clump(tmp_path):
    rows = [
        {"name": I, "mag": 15.0, "err": 0.05, "photType": {"pos": ["Source"]}}
    ]
    system = _prepared(
        tmp_path,
        rows=rows,
        clump={"name": "RedClump", "bands": ["I"]},
        params=_params(
            **{"redclump.RedClump.extinction_0": {"mu": 1.8, "sigma": 0.05}}
        ),
    )
    assert system.redclump.manifest["extinction"]["shape"] == (1,)


def test_a_source_named_like_a_star_raises(tmp_path):
    with pytest.raises(ValueError, match="name of a star"):
        _prepared(tmp_path, clump={"name": "Source"}, rows=_rows()[:1])


# ---------------------------------------------------------------------------
# The seam: a source is evaluated exactly like a star
# ---------------------------------------------------------------------------
def test_a_flux_source_twin_of_a_star_predicts_it_bit_for_bit(
    tmp_path, monkeypatch
):
    """
    Given a test double whose flux source reads star A's own parameters,
    When the forward model is built,
    Then its predicted magnitudes equal A's to the last bit; an index in a
      photType never resolves to it.
    """
    paths = {
        k: f"star.0.{k}"
        for k in ("teffsed", "loggsed", "feh", "av", "radiussed", "distance")
    }
    monkeypatch.setattr(
        Star,
        "sed_flux_sources",
        lambda self, system: [SEDFluxSource("Twin", paths, 4)],
    )
    sed_path = tmp_path / "twin.sed"
    sed_path.write_text(
        yaml.safe_dump(
            {
                "model": "NextGen",
                "filters": [
                    {
                        "name": n,
                        "mag": 10.0,
                        "err": 0.05,
                        "photType": {"pos": [p]},
                    }
                    for n, p in (
                        ("2MASS/2MASS.J", "A"),
                        ("2MASS/2MASS.H", "Twin"),
                        ("2MASS/2MASS.Ks", 0),
                        ("GAIA/GAIA3.G", "A"),
                        ("2MASS/2MASS.J", "Twin"),
                        ("2MASS/2MASS.H", "A"),
                    )
                ],
            }
        )
    )
    system = System(
        {
            "star": [{"name": "A", "mist": False}],
            "sed": {"file": str(sed_path)},
        },
        {},
    )
    system.prepare()
    sed = system.sed
    assert sed.pos_matrix[:, 1].tolist() == [0, 1, 0, 0, 1, 0]
    assert sed.pos_matrix[2].tolist() == [1, 0]  # index 0 is star A
    model = system.build_model()
    with model:
        m = sed._predicted_appmag_node(system)
    got = _at(model, m, model.initial_point())
    np.testing.assert_array_equal(got[0], got[1])


# ---------------------------------------------------------------------------
# The soft av tie
# ---------------------------------------------------------------------------
def _tied(tmp_path, monkeypatch, flag=None, av=None):
    monkeypatch.setattr(Star, "selected_sources", lambda self, system: [0])
    clump = {"name": "RedClump", "bands": ["I"]}
    if flag is not None:
        clump["clump_constrains_source_av"] = flag
    extra = {} if av is None else {"redclump.av": {"initval": av}}
    return _prepared(tmp_path, clump=clump, params=_params(**extra))


def test_the_tie_is_the_derived_gaussian_and_the_flag_removes_it(
    tmp_path, monkeypatch
):
    """
    Given a selected source whose SED flux is read,
    When the clump ties it,
    Then av_ratio exists on that star alone, its potential is exactly
      -(r - 1)^2 / (2 0.09^2) (unnormalized, constant sigma), and false
      removes it with nothing else in the sampled set changing.
    """
    on = _tied(tmp_path, monkeypatch)
    assert on.redclump.tied == [0]
    m_on = on.build_model()
    off = _tied(tmp_path, monkeypatch, flag=False)
    assert "av_ratio" not in off.redclump.manifest
    m_off = off.build_model()
    ip = m_on.initial_point()
    assert set(ip) == set(m_off.initial_point())
    tie = [
        p
        for p in m_on.potentials
        if p.name == "gaussian_prior.redclump.av_ratio"
    ]
    assert len(tie) == 1
    assert not any("av_ratio" in p.name for p in m_off.potentials)
    r, term = (
        float(np.atleast_1d(x).sum())
        for x in _at(m_on, [on.redclump.av_ratio.value, tie[0]], ip)
    )
    assert term == pytest.approx(-((r - 1) ** 2) / (2 * 0.09**2), abs=1e-8)


def test_explicit_true_with_nothing_to_tie_raises(tmp_path):
    with pytest.raises(ValueError, match="clump_constrains_source_av: true"):
        _prepared(
            tmp_path,
            clump={"name": "RedClump", "clump_constrains_source_av": True},
        )


def test_the_tie_gradient_is_finite_at_the_av_floor(tmp_path, monkeypatch):
    system = _tied(tmp_path, monkeypatch, av=0.0501)
    model = system.build_model()
    ip = model.initial_point()
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))
    jax = pytest.importorskip("jax")
    _ = jax
    assert np.all(np.isfinite(model.compile_dlogp(mode="JAX")(ip)))


def _at(model, outs, point):
    """Evaluate ``outs`` at a point dict of the model's value variables."""
    import pytensor

    fn = pytensor.function(model.value_vars, outs, on_unused_input="ignore")
    return fn(*[point[v.name] for v in model.value_vars])
