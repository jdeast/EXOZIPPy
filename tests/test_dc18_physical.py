"""DC2018 physical-quantity scoring (review 7.16.5): bias and coverage.

examples/DC2018/dc18_common.py scores the lens mass, D_L, D_S, |mu_rel| and
its direction, the planet mass and a_perp as the truth's percentile in the
posterior draws (PIT).  These tests pin the arithmetic that a wrong frame or
unit would silently break: the galactic -> equatorial rotation of the key's
proper motions (against astropy), the key parsing (on a synthetic answer key
carrying event 128's real values, so the DC2018 data tree is not needed),
the PIT itself on the circle, and the trace reader's burn-in, chain filter
and a_perp = s * theta_E * D_L.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

DC18 = Path(__file__).resolve().parents[1] / "examples" / "DC2018"

# Event 128 of master_file.txt (also in examples/DC2018/dc128_truth_forward.py)
K128 = dict(
    ra=267.595,
    dec=-28.982,
    Ds=8.143,
    smu_l=-7.73446,
    smu_b=-2.6885,
    Dl=7.999,
    Ml=0.454,
    lmu_l=-6.48044,
    lmu_b=-1.37793,
    thE=0.0904762,
    murel=1.81388,
    q=0.0012118,
    s=0.993145,
)


@pytest.fixture(scope="module")
def common():
    sys.path.insert(0, str(DC18))
    try:
        yield importlib.import_module("dc18_common")
    finally:
        sys.path.remove(str(DC18))


def _astropy_pm(mu_l, mu_b, ra, dec):
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    g = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    gg = SkyCoord(
        l=g.l,
        b=g.b,
        distance=1.0 * u.kpc,
        frame="galactic",
        pm_l_cosb=mu_l * u.mas / u.yr,
        pm_b=mu_b * u.mas / u.yr,
    ).icrs
    return (
        float(gg.pm_ra_cosdec.to_value(u.mas / u.yr)),
        float(gg.pm_dec.to_value(u.mas / u.yr)),
    )


@pytest.mark.parametrize(
    "ra,dec,mu_l,mu_b",
    [
        (267.595, -28.982, 1.25, 1.31),
        (269.165, -29.0207, 8.85, 4.30),
        (271.0, -33.5, -3.0, 0.4),
        (10.0, 41.0, 2.0, -5.0),
    ],
)
def test_galactic_pm_rotation_matches_astropy(common, ra, dec, mu_l, mu_b):
    """Given a galactic proper motion at a position,
    When it is rotated onto the local (E, N) axes,
    Then it matches astropy's transformation to 1e-6 mas/yr."""
    got = common.galactic_pm_to_equatorial(mu_l, mu_b, ra, dec)
    want = _astropy_pm(mu_l, mu_b, ra, dec)
    assert np.allclose(got, want, atol=1e-6)


@pytest.fixture()
def key_tree(tmp_path):
    """A minimal answer key: event 1 carries event 128's real values."""
    names = ["idx", *K128, "rE", "Mp"]
    ans = tmp_path / "Answers"
    ans.mkdir()
    (ans / "wfirstColumnNumbers.txt").write_text(
        "#colno name\n\n" + "".join(f"{i} {n}\n" for i, n in enumerate(names))
    )
    vals = dict(K128, rE=K128["thE"] * K128["Dl"], Mp=K128["q"] * K128["Ml"])
    row = " ".join(["1"] + [repr(vals[n]) for n in names[1:]])
    (ans / "master_file.txt").write_text(
        "# header\n" + row + " omcassan_0_1_1 1\n"
    )
    return tmp_path


def test_physical_truth_units_and_direction(common, key_tree):
    """Given the key's galactic proper motions, distances and masses,
    When the physical truths are built,
    Then the units are Msun / kpc / mas/yr / Mjup / AU, a_perp is s * rE,
    and the direction is the PA (E of N) of lens-minus-source, built
    independently here with astropy -- with no consistency note raised."""
    truth, notes = common.physical_truth(key_tree, 1)
    assert truth["M_lens"] == K128["Ml"]
    assert truth["D_L"] == K128["Dl"] and truth["D_S"] == K128["Ds"]
    assert truth["mu_rel"] == K128["murel"]
    assert truth["M_planet"] == pytest.approx(
        K128["q"] * K128["Ml"] / 9.5458e-4, rel=1e-12
    )
    assert truth["a_perp"] == pytest.approx(
        K128["s"] * K128["thE"] * K128["Dl"], rel=1e-12
    )
    e, n = _astropy_pm(
        K128["lmu_l"] - K128["smu_l"],
        K128["lmu_b"] - K128["smu_b"],
        K128["ra"],
        K128["dec"],
    )
    assert truth["phi_mu"] == pytest.approx(
        np.degrees(np.arctan2(e, n)) % 360.0, abs=1e-5
    )
    assert not any(notes.values())


def test_physical_truth_flags_a_frame_mismatch(common, key_tree):
    """Given a key whose murel is NOT |lmu - smu|,
    When the physical truths are built,
    Then the mu_rel and phi_mu rows carry a note instead of passing."""
    f = key_tree / "Answers" / "master_file.txt"
    f.write_text(f.read_text().replace(repr(K128["murel"]), "2.5"))
    _, notes = common.physical_truth(key_tree, 1)
    assert notes["mu_rel"] and notes["phi_mu"]
    assert not notes["M_lens"]


def test_score_draws_pit_and_coverage(common):
    """Given Gaussian draws N(10, 1),
    When the truth sits at the median, at +1 sigma and at +3 sigma,
    Then the PIT is 0.5, ~0.841 and ~0.9987, the 68% coverage flips at
    1 sigma, and z is negative when the posterior is BELOW the truth."""
    d = np.random.default_rng(1).normal(10.0, 1.0, 400_000)
    at = common.score_draws(d, 10.0)
    assert at["pit"] == pytest.approx(0.5, abs=0.003)
    assert at["in68"] and at["in95"]
    one = common.score_draws(d, 11.0)
    assert one["pit"] == pytest.approx(0.8413, abs=0.003)
    assert one["z"] == pytest.approx(-1.0, abs=0.02)
    three = common.score_draws(d, 13.0)
    assert not three["in68"] and not three["in95"]
    assert three["log10_ratio"] == pytest.approx(
        np.log10(10.0 / 13.0), abs=1e-3
    )


def test_score_draws_angle_unwraps_across_zero(common):
    """Given angle draws straddling 0/360 (centered on 0 deg, sd 5),
    When the truth is 5 deg,
    Then the PIT is the +1 sigma value, not a wrap artifact."""
    d = np.random.default_rng(2).normal(0.0, 5.0, 400_000) % 360.0
    sc = common.score_draws(d, 5.0, kind="angle")
    assert sc["pit"] == pytest.approx(0.8413, abs=0.003)
    assert sc["bias"] == pytest.approx(-5.0, abs=0.05)
    assert sc["log10_ratio"] is None


def test_ensemble_detects_one_directional_bias(common):
    """Given PITs uniform on (0, 1) and PITs all below 0.3,
    When the ensemble is summarized,
    Then the uniform set is consistent with calibration and the low set is
    flagged one-directional (every posterior median ABOVE the truth)."""
    uni = [
        {
            "param": "M_lens",
            "pit": p,
            "in68": 0.159 <= p <= 0.841,
            "in95": 0.025 <= p <= 0.975,
            "bias": 0.0,
            "log10_ratio": 0.0,
        }
        for p in (np.arange(40) + 0.5) / 40
    ]
    low = [dict(r, pit=0.3 * r["pit"]) for r in uni]
    eu = common.ensemble_physical(uni)["M_lens"]
    el = common.ensemble_physical(low)["M_lens"]
    assert eu["ks_p"] > 0.5 and eu["sign_p"] > 0.5
    assert el["n_high"] == 40 and el["sign_p"] < 1e-10 and el["ks_p"] < 1e-6
    assert sum(eu["hist"]) == 40


def test_physical_draws_reads_the_pipelines_draws(common, tmp_path):
    """Given a synthetic trace of 4 chains x 10 draws, one chain stuck far
    below the others' lp, and a wrap-up header saying burn-in 4 / 3 chains,
    When the physical draws are read,
    Then the stuck chain and the first 4 draws are dropped, the Lens /
    Source / companion elements follow the config's block order, and
    a_perp is s * theta_E * D_L[kpc] per draw."""
    import xarray as xr
    import yaml

    nc, nd = 4, 10
    rng = np.random.default_rng(3)
    lp = rng.normal(100.0, 1.0, (nc, nd))
    lp[2] -= 1000.0  # chain 2 never reaches the good region
    # star order SOURCE first, to prove the config is read
    mass = np.stack([np.full((nc, nd), 0.9), np.full((nc, nd), 0.3)], -1)
    dist = np.stack([np.full((nc, nd), 8000.0), np.full((nc, nd), 4000.0)], -1)
    s = np.stack([np.zeros((nc, nd)), np.full((nc, nd), 1.5)], -1)
    mass[:, :4, 1] = 99.0  # burn-in values that must not survive
    post = xr.Dataset(
        {
            "star.mass": (("chain", "draw", "s0"), mass),
            "star.distance": (("chain", "draw", "s0"), dist),
            "lens.s": (("chain", "draw", "l0"), s),
            "mulensevent.theta_E": (
                ("chain", "draw", "x"),
                np.full((nc, nd, 1), 0.2),
            ),
            "mulensevent.mu_ra_rel": (
                ("chain", "draw", "x"),
                np.full((nc, nd, 1), 3.0),
            ),
            "mulensevent.mu_dec_rel": (
                ("chain", "draw", "x"),
                np.full((nc, nd, 1), -4.0),
            ),
            "planet.mass": (("chain", "draw", "x"), np.full((nc, nd, 1), 0.5)),
        }
    )
    trace = tmp_path / "X_trace.nc"
    post.to_netcdf(trace, group="posterior")
    xr.Dataset({"lp": (("chain", "draw"), lp)}).to_netcdf(
        trace, group="sample_stats", mode="a"
    )
    (tmp_path / "X_summary.txt").write_text(
        "# burn-in discarded: 4 draws (40% of 10); chains kept: 3\n"
    )
    (tmp_path / "X.yaml").write_text(
        yaml.safe_dump(
            {
                "star": [{"name": "Source"}, {"name": "Lens"}],
                "lens": [{"body": "star.Lens"}, {"body": "planet.Companion"}],
            }
        )
    )
    draws, info = common.physical_draws(
        trace, tmp_path / "X.yaml", tmp_path / "X_summary.txt"
    )
    assert info["n_chains"] == 3 and info["burnin"] == 4
    assert info["n_draws_kept"] == 18 and not info["warnings"]
    assert np.all(draws["M_lens"] == 0.3)
    assert np.all(draws["D_L"] == 4.0) and np.all(draws["D_S"] == 8.0)
    assert np.allclose(draws["a_perp"], 1.5 * 0.2 * 4.0)
    assert np.allclose(draws["mu_rel"], 5.0)  # rebuilt from the components
    assert np.allclose(draws["phi_mu"], np.degrees(np.arctan2(3.0, -4.0)))
    assert draws["M_lens"].size == 18


def test_physical_draws_falls_back_to_log10_coordinates(common, tmp_path):
    """Given an older trace that stores only star.logmass, lens.log_s and
    mulensevent.log_theta_E (the d228e190 static30 layout, 2-D scalars),
    When the physical draws are read with no config or summary,
    Then mass, s and theta_E are rebuilt as 10**log and a_perp follows."""
    import xarray as xr

    nc, nd = 3, 8
    lp = np.random.default_rng(4).normal(0.0, 1.0, (nc, nd))
    post = xr.Dataset(
        {
            "star.logmass": (
                ("chain", "draw", "s0"),
                np.stack([np.full((nc, nd), -0.5), np.zeros((nc, nd))], -1),
            ),
            "star.distance": (
                ("chain", "draw", "s0"),
                np.stack(
                    [np.full((nc, nd), 6000.0), np.full((nc, nd), 8000.0)], -1
                ),
            ),
            "lens.log_s": (
                ("chain", "draw", "l0"),
                np.stack([np.zeros((nc, nd)), np.full((nc, nd), 0.25)], -1),
            ),
            "mulensevent.log_theta_E": (
                ("chain", "draw"),
                np.full((nc, nd), -1.0),
            ),
        }
    )
    trace = tmp_path / "Y_trace.nc"
    post.to_netcdf(trace, group="posterior")
    xr.Dataset({"lp": (("chain", "draw"), lp)}).to_netcdf(
        trace, group="sample_stats", mode="a"
    )
    draws, info = common.physical_draws(trace)
    assert info["burnin"] == 4 and "NO WRAP-UP HEADER" in info["provenance"]
    assert np.allclose(draws["M_lens"], 10**-0.5)
    assert np.allclose(draws["a_perp"], 10**0.25 * 0.1 * 6.0)
    assert any("M_planet: not in trace" in w for w in info["warnings"])
