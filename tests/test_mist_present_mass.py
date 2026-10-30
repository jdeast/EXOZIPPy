"""The MIST grid on the PRESENT-DAY mass (components/evolutionarymodel).

``star.logmass`` is the mass a star has now; MIST tabulates a track per
INITIAL mass and carries the present-day ``star_mass`` on every row.
``mist_grid._reaxis_to_present_mass`` inverts it once at load, so the
likelihood reads the tracks at the initial mass the sampled present-day
mass implies, and the component adds the Jacobian that keeps the prior a
density in the initial mass (``evolutionarymodel.md``).

The synthetic-grid tests pin the algebra exactly.  The ``real_grid`` tests
read the published MISTv2.5 grid and skip where it is not on disk; their
reference numbers are quoted from the raw MIST v2.5 EEP tracks
(https://mist.science/data/tarballs_v2.5/eeps/, [a/Fe] = 0, v/vcrit = 0).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest

from exozippy.components.evolutionarymodel import mist_grid
from exozippy.components.evolutionarymodel.evolutionarymodel import (
    MASS_JACOBIAN_MAX,
    MASS_JACOBIAN_MIN,
)
from exozippy.components.galacticmodel.galacticmodel import GalacticModel

OUT = mist_grid.OUTPUT_INDEX
TRK = mist_grid.TRACK_INDEX
TABULATED = (
    "feh_mist",
    "radius_mist",
    "teff_mist",
    "age_mist_gyr",
    "dEEP_dage",
    "here_be_dragons",
)


def _grid(tmp_path, mass_loss=None, masses=(0.5, 1.0, 2.0)):
    from conftest import write_synthetic_mist_grid

    root = write_synthetic_mist_grid(
        tmp_path, masses=masses, mass_loss=mass_loss
    )
    mist_grid._GRID_CACHE.clear()
    return mist_grid.load_mist_grid(model_root=root)


# ----------------------------------------------------------------------
# The re-axing, on a synthetic grid
# ----------------------------------------------------------------------


def test_without_mass_loss_the_present_day_grid_is_the_tabulated_one(
    tmp_path,
):
    """
    Given a grid whose present-day mass equals its initial mass on every row,
    When it is loaded,
    Then the re-axed array IS the tabulated one, the initial mass is the
      axis itself and the Jacobian is exactly 1.
    """
    # Act
    grid = _grid(tmp_path)

    # Assert
    for c in TABULATED:
        np.testing.assert_array_equal(
            grid["values"][..., OUT[c]], grid["track_values"][..., TRK[c]]
        )
    np.testing.assert_allclose(
        grid["values"][..., OUT["logmass_initial"]],
        np.broadcast_to(
            grid["logmass_pts"][:, None, None], grid["values"].shape[:3]
        ),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        grid["values"][..., OUT["dlogmass_initial_dlogmass"]], 1.0, rtol=1e-14
    )
    assert grid["reaxis"] == {"n_unreachable": 0, "n_hidden": 0}


def test_a_present_day_mass_reads_the_track_of_the_initial_mass_it_implies(
    tmp_path,
):
    """
    Given tracks that lose 5% of their mass by the last EEP, the same
      fraction at every mass,
    When the 1 solMass present-day node is read,
    Then it is the track of initial mass 1/(1 - loss) at each EEP --
      exactly what interpolating the TABULATED tracks there gives -- and
      d log M_ini / d log M is 1.
    """
    # Arrange
    grid = _grid(tmp_path, mass_loss=lambda m, frac: 0.05 * frac)
    i = int(np.flatnonzero(np.isclose(grid["logmass_pts"], 0.0))[0])
    eeps = grid["eep_pts"]
    frac = np.arange(eeps.size) / (eeps.size - 1)
    x_expected = -np.log10(1.0 - 0.05 * frac)

    for j, feh in enumerate(grid["initfeh_pts"]):
        row = grid["values"][i, j]

        # Assert: the implied initial mass
        np.testing.assert_allclose(
            row[:, OUT["logmass_initial"]], x_expected, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            row[:, OUT["dlogmass_initial_dlogmass"]], 1.0, rtol=1e-9
        )
        # ... and the tabulated tracks interpolated there, EEP by EEP
        for k in range(eeps.size):
            track = mist_grid.interpolate_track(
                grid, x_expected[k], feh, mass="initial"
            )
            for c in TABULATED:
                assert row[k, OUT[c]] == pytest.approx(
                    track[k, TRK[c]], rel=1e-10, abs=1e-12
                )


def test_the_jacobian_is_the_cells_slope_of_initial_on_present_mass(tmp_path):
    """
    Given a loss that grows with mass (so the present-day mass is a
      nonlinear function of the initial one),
    Then each node's d log M_ini / d log M is its cell's
      (x_hi - x_lo) / (y_hi - y_lo).
    """
    # Arrange
    grid = _grid(tmp_path, mass_loss=lambda m, frac: 0.1 * m * frac)
    x = grid["logmass_pts"]
    y = grid["track_values"][..., TRK["logmass_present"]]
    k = -1  # the last EEP, where the loss is largest
    j = 0

    # Assert: the 1 solMass present-day node sits in the (1, 2) cell
    # (1 solMass initial is now 0.9), so its slope is that cell's.
    i = int(np.flatnonzero(np.isclose(x, 0.0))[0])
    slope = (x[i + 1] - x[i]) / (y[i + 1, j, k] - y[i, j, k])
    assert grid["values"][i, j, k, OUT["dlogmass_initial_dlogmass"]] == (
        pytest.approx(slope, rel=1e-12)
    )


def test_a_present_day_mass_no_track_still_has_is_a_dragon(tmp_path):
    """
    Given tracks that lose mass, so the heaviest is lighter now than the
      top of the axis,
    When the top present-day node is read late on the track,
    Then it is held at the heaviest track and flagged with here_be_dragons
      by UNREACHABLE_DRAGONS_PER_DEX per dex out of reach (at least 1);
      the first EEP, before any loss, is not flagged.
    """
    # Arrange
    grid = _grid(tmp_path, mass_loss=lambda m, frac: 0.2 * frac)
    beyond = -np.log10(1.0 - 0.2)

    # Act
    dragons = grid["values"][-1, :, :, OUT["here_be_dragons"]]

    # Assert
    assert dragons[:, -1] == pytest.approx(
        max(1.0, mist_grid.UNREACHABLE_DRAGONS_PER_DEX * beyond)
    )
    assert np.all(dragons[:, 0] == 0.0)
    assert grid["values"][-1, 0, -1, OUT["logmass_initial"]] == (
        pytest.approx(grid["logmass_pts"][-1])
    )
    assert grid["reaxis"]["n_unreachable"] > 0


def test_a_track_hidden_by_a_lighter_ones_mass_loss_is_flagged(tmp_path):
    """
    Given a 2 solMass track that ends lighter (0.8 now) than the 1 solMass
      track (no loss), between a 1 and a 4 solMass track that lose nothing --
      present-day mass NOT increasing in initial mass,
    When the grid is re-axed,
    Then the inverse is taken on the envelope (1 solMass now IS the 1
      solMass track), and the 2 solMass present-day node, blended across
      the hidden track, is flagged; before any loss it is not.
    """
    # Arrange / Act
    grid = _grid(
        tmp_path,
        masses=(0.5, 1.0, 2.0, 4.0),
        mass_loss=lambda m, frac: 0.6 * frac if m == 2.0 else 0.0,
    )
    x = grid["logmass_pts"]
    i1 = int(np.flatnonzero(np.isclose(x, 0.0))[0])
    i2 = int(np.flatnonzero(np.isclose(x, np.log10(2.0)))[0])

    # Assert
    assert grid["values"][i1, 0, -1, OUT["logmass_initial"]] == pytest.approx(
        0.0, abs=1e-12
    )
    assert grid["values"][i2, 0, -1, OUT["here_be_dragons"]] >= 1.0
    assert grid["values"][i2, 0, 0, OUT["here_be_dragons"]] == 0.0
    assert grid["reaxis"]["n_hidden"] > 0


def test_a_grid_without_the_present_day_mass_raises(tmp_path):
    """
    Given a grid built before the star_mass column existed,
    When it is assembled,
    Then it raises naming the column -- reading it as "no mass loss" would
      silently restore the initial-mass assumption.
    """
    # Arrange
    rows = [
        {
            "mass": m,
            "initfeh": f,
            "EEP": e,
            "feh_mist": f,
            "radius_mist": 1.0,
            "teff_mist": 5000.0,
            "age_mist": 1e9 * e,
            "dEEP_dage": 1e-9,
            "here_be_dragons": 0.0,
        }
        for m in (1.0, 2.0)
        for f in (0.0, 0.5)
        for e in (1, 2)
    ]

    # Act / Assert
    with pytest.raises(ValueError, match="star_mass"):
        mist_grid._assemble_grid(pd.DataFrame(rows))


# ----------------------------------------------------------------------
# The component: the Jacobian, the initial mass, and the IMF seam
# ----------------------------------------------------------------------


def _built_with_loss(tmp_path, loss):
    from conftest import write_synthetic_mist_grid
    from exozippy.system import System

    root = write_synthetic_mist_grid(tmp_path, mass_loss=loss)
    mist_grid._GRID_CACHE.clear()
    config = {
        "sampler": {"draws": 10},
        "star": [{"name": "A"}],
        "evolutionarymodel": [{"star": "A", "model_root": root}],
    }
    system = System(config, {})
    system.prepare()
    return system, system.build_model()


def test_the_initial_mass_and_its_jacobian_reach_the_model(tmp_path):
    """
    Given a star on a grid whose tracks lose 5% of their mass by the end,
    When the model is built,
    Then the initial mass is a Deterministic, initial_logmass hands the
      same node to a mass function, and the mass Jacobian is a potential
      equal to log(d log M_ini / d log M), finite at the start.
    """
    # Arrange
    system, model = _built_with_loss(tmp_path, lambda m, frac: 0.05 * frac)
    comp = system.active_components["evolutionarymodel"]
    point = model.initial_point()

    def at_start(node):
        import pytensor

        [node] = model.replace_rvs_by_values([node])
        f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
        return f(*[point[v.name] for v in model.value_vars])

    # Act
    m_ini = at_start(model["evolutionarymodel.mass_initial"])
    claims = comp.initial_logmass(system)
    x = at_start(claims[0])
    pred = at_start(comp._interpolate(system))
    jac = pred[0, OUT["dlogmass_initial_dlogmass"]]
    potentials = {p.name: p for p in model.potentials}
    mass_jac = at_start(potentials["evolutionarymodel.mass_jacobian"])

    # Assert
    assert list(claims) == [0]
    assert float(m_ini[0]) == pytest.approx(10.0 ** float(x))
    assert float(mass_jac) == pytest.approx(
        np.log(np.clip(jac, MASS_JACOBIAN_MIN, MASS_JACOBIAN_MAX))
    )
    assert np.isfinite(model.compile_logp()(point))


def _imf_system(claims_by_comp, logmass):
    comps = {
        name: SimpleNamespace(
            prefix=name, initial_logmass=lambda system, c=c: c
        )
        for name, c in claims_by_comp.items()
    }
    stars = SimpleNamespace(
        logmass=SimpleNamespace(value=pt.as_tensor_variable(logmass))
    )
    return SimpleNamespace(active_components=comps), stars


def test_the_imf_reads_the_birth_mass_a_component_supplies():
    """
    Given three stars, one of which a component says was born at
      log M = 0.1 while it is 0.0 now,
    When galacticmodel assembles the mass its IMF is evaluated at,
    Then that star's entry is its birth mass and the others are untouched.
    """
    # Arrange
    system, stars = _imf_system(
        {"evolutionarymodel": {1: pt.as_tensor_variable(0.1)}},
        np.array([-0.5, 0.0, 0.3]),
    )
    gm = GalacticModel.__new__(GalacticModel)

    # Act
    birth = gm._birth_logmass(system, stars).eval()

    # Assert
    np.testing.assert_allclose(birth, [-0.5, 0.1, 0.3])


def test_two_components_claiming_one_birth_mass_raise():
    """
    Given two components each supplying an initial mass for star 0,
    Then galacticmodel raises naming both: a star has one birth mass.
    """
    # Arrange
    system, stars = _imf_system(
        {
            "first": {0: pt.as_tensor_variable(0.1)},
            "second": {0: pt.as_tensor_variable(0.2)},
        },
        np.array([0.0]),
    )
    gm = GalacticModel.__new__(GalacticModel)

    # Act / Assert
    with pytest.raises(ValueError, match="first.*second"):
        gm._birth_logmass(system, stars)


# ----------------------------------------------------------------------
# The published grid (skipped where it is not on disk)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_grid():
    path = mist_grid.grid_path()
    if not path.is_file():
        pytest.skip(f"the published MIST grid is not on disk at {path}")
    df = pd.read_parquet(path)
    if mist_grid.PRESENT_MASS_COLUMN not in df.columns:
        pytest.skip(f"{path} predates the star_mass column")
    return df


# (initfeh, initial mass, EEP) -> star_mass, quoted from the raw MIST v2.5
# tracks feh_{p000,m100,p025}_afe_p0_vvcrit0.0/eeps/<mass>M.track.eep.
RAW_STAR_MASS = {
    (0.0, 1.0, 202): 0.9999975861218052,
    (0.0, 1.0, 454): 0.9994175712108893,
    (0.0, 1.0, 631): 0.95796969254209,
    (0.0, 1.0, 707): 0.9556476170811179,
    (0.0, 1.0, 807): 0.9539340041117546,
    (0.0, 0.8, 670): 0.7414682960815171,
    (-1.0, 1.5, 670): 1.4831852194274258,
    (0.25, 20.0, 670): 14.159684479401392,
}


@pytest.mark.parametrize("key", sorted(RAW_STAR_MASS))
def test_the_star_mass_column_is_the_raw_tracks(real_grid, key):
    """
    Given a point of the published grid,
    Then its star_mass is the raw MIST track's, to the last bit or two:
      the generator parses the tracks with pandas' python engine, whose
      float parser is not correctly rounded (1.0 solMass, EEP 202 reads
      0.9999975861218051 for the text 9.9999758612180523E-001), the same
      parser every other column of the grid went through.
    """
    feh, mass, eep = key
    row = real_grid[
        (real_grid.initfeh == feh)
        & np.isclose(real_grid.mass, mass)
        & (real_grid.EEP == eep)
    ]
    assert len(row) == 1
    assert float(row.star_mass.iloc[0]) == pytest.approx(
        RAW_STAR_MASS[key], rel=5e-16, abs=0.0
    )


def test_a_red_clump_star_is_lighter_now_by_the_tabulated_amount(real_grid):
    """
    Given the solar-metallicity 1 solMass track in the core-helium-burning
      clump (EEP 670),
    When the loader re-axes the grid,
    Then that star is 4.2% lighter now (0.95797 solMass), and the
      present-day mass it has now reads back the 1 solMass INITIAL track:
      its implied initial mass is 1.000 and its radius and Teff are the
      tabulated ones to the interpolation's precision.
    """
    # Arrange
    mist_grid._GRID_CACHE.clear()
    grid = mist_grid.load_mist_grid()
    j = int(np.flatnonzero(grid["initfeh_pts"] == 0.0)[0])
    k = int(np.flatnonzero(grid["eep_pts"] == 670.0)[0])
    m_now = RAW_STAR_MASS[(0.0, 1.0, 631)]  # the clump plateau

    # Act
    present = mist_grid.interpolate_track(grid, np.log10(m_now), 0.0)[k]
    initial = mist_grid.interpolate_track(grid, 0.0, 0.0, mass="initial")[k]

    # Assert
    assert 1.0 - 10.0 ** initial[TRK["logmass_present"]] == pytest.approx(
        0.0420, abs=5e-4
    )
    assert 10.0 ** present[OUT["logmass_initial"]] == pytest.approx(
        1.0, abs=2e-3
    )
    for c in ("radius_mist", "teff_mist"):
        assert present[OUT[c]] == pytest.approx(initial[TRK[c]], rel=1e-2)
    assert j >= 0


def test_main_sequence_stars_are_unchanged(real_grid):
    """
    Given every main-sequence row (ZAMS EEP 202 to TAMS 454) of the
      published grid below 5 solMass that a star can reach in 13.8 Gyr and
      that no here_be_dragons flag marks,
    When the grid is re-axed on the present-day mass,
    Then the initial mass a present-day mass implies differs from it by
      < 1e-3 (fractional), and radius, Teff and age by < 1% and [Fe/H] by
      < 0.01 dex -- a third of the 3% systematic floor or less. (They move
      by more than the mass does where they are steep in mass: radius 0.4%
      and age 0.7% at the 1.13-1.15 solMass convective-core hook.)
    """
    # Arrange
    mist_grid._GRID_CACHE.clear()
    grid = mist_grid.load_mist_grid()
    eep = grid["eep_pts"]
    x = grid["logmass_pts"]
    v, t = grid["values"], grid["track_values"]
    sel = (
        ((eep >= 202) & (eep <= 454))[None, None, :]
        & (x < np.log10(5.0))[:, None, None]
        & (t[..., TRK["here_be_dragons"]] <= 0)
        & (v[..., OUT["here_be_dragons"]] <= 0)
        & (t[..., TRK["age_mist_gyr"]] < 13.8)
    )

    # Act
    m_ratio = 10.0 ** (v[..., OUT["logmass_initial"]] - x[:, None, None])

    # Assert
    assert sel.sum() > 100000
    assert np.abs(m_ratio[sel] - 1.0).max() < 1e-3
    for c in ("radius_mist", "teff_mist", "age_mist_gyr"):
        rel = np.abs(v[..., OUT[c]] / t[..., TRK[c]] - 1.0)[sel]
        assert rel.max() < 1e-2, c
    dfeh = np.abs(v[..., OUT["feh_mist"]] - t[..., TRK["feh_mist"]])[sel]
    assert dfeh.max() < 1e-2
