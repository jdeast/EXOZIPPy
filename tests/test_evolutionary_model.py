"""Contract tests for components/evolutionarymodel.

Three things are pinned here, and each one is a bug that shipped in the
first draft of this component:

* ``mist_grid``'s completeness check must tolerate a table with one row per
  (mass, initfeh, EEP).  It was written as ``MultiIndex.equals``, which is
  order- and duplicate-sensitive and therefore False for EVERY grid,
  complete or not -- so ``trim_to_complete_grid`` raised unconditionally and
  the component could not load at all.
* The component declares NO parameters of its own.  ``star.initfeh`` and
  ``star.eep`` already exist and are materialized by the
  ``in_system("evolutionarymodel")`` branch of ``Star.register_parameters``;
  a second copy on this component's own block would leave the star's
  materialized, free and read by nothing.
* The EEP -> age Jacobian is ``-log|dEEP/dage|`` with a two-sided CLIP.  A
  bare floor there is an unbounded reward (``-log(1e-30)`` = +69 nats), not
  a safety rail -- the opposite of ``calc_jitter``'s radicand floor.
"""

import logging

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest

from exozippy.components.evolutionarymodel import mist_grid, physics
from exozippy.components.evolutionarymodel.evolutionarymodel import (
    CONSTRAINABLE,
    DEEP_DAGE_MAX,
    DEEP_DAGE_MIN,
    KIEL_EEP_WINDOW,
    KIEL_X_PAD_FRAC,
    KIEL_INDEX,
    KIEL_LOGG_WINDOW,
    KIEL_WINDOW_MARGIN_FRAC,
    EvolutionaryModel,
)
from exozippy.components.relations import StellarRelation

EEPS = (1, 2, 3)


def _track_table(pairs, eeps=EEPS, **overrides):
    """A per-(mass, initfeh, EEP)-row grid table, the shape the parquet has.

    Every column the loader reads is present; values are arbitrary but
    distinct so a mis-indexed reshape shows up.
    """
    rows = []
    for mass, feh in pairs:
        for k, eep in enumerate(eeps):
            rows.append(
                {
                    "mass": float(mass),
                    "initfeh": float(feh),
                    "EEP": int(eep),
                    "feh_mist": float(feh) + 0.01 * k,
                    "radius_mist": 1.0 + 0.1 * k,
                    "teff_mist": 5000.0 + 100.0 * k,
                    "age_mist": 1.0e9 * (k + 1),
                    "dEEP_dage": 1.0e-9 * (k + 1),
                    "here_be_dragons": 0.0,
                }
            )
    df = pd.DataFrame(rows)
    for col, val in overrides.items():
        df[col] = val
    return df


# ----------------------------------------------------------------------
# Completeness and trimming
# ----------------------------------------------------------------------


def test_a_complete_grid_with_per_eep_rows_reads_as_complete():
    """
    Given a complete (mass, initfeh) rectangle tabulated once per EEP,
    When the completeness check runs,
    Then it says complete -- the EEP duplicates are not holes.
    """
    # Arrange
    df = _track_table([(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 0.5)])

    # Act / Assert
    assert mist_grid._check_complete_grid(df) is True


def test_a_holed_grid_reads_as_incomplete():
    """
    Given a (mass, initfeh) rectangle with one missing track,
    When the completeness check runs,
    Then it says incomplete.
    """
    # Arrange
    df = _track_table([(1.0, 0.0), (1.0, 0.5), (2.0, 0.0)])

    # Act / Assert
    assert mist_grid._check_complete_grid(df) is False


def test_trimming_a_holed_grid_yields_a_complete_one():
    """
    Given a grid missing the (2.0, -1.0) track,
    When it is trimmed along the metallicity axis,
    Then the -1.0 column is gone and the result is complete.
    """
    # Arrange
    df = _track_table(
        [(1.0, -1.0), (1.0, 0.0), (2.0, 0.0), (3.0, -1.0), (3.0, 0.0)]
    )

    # Act
    trimmed, dropped = mist_grid.trim_to_complete_grid(df)

    # Assert
    assert dropped == {"dropped_mass": [], "dropped_initfeh": [-1.0]}
    assert mist_grid._check_complete_grid(trimmed)
    assert sorted(trimmed["initfeh"].unique()) == [0.0]
    assert sorted(trimmed["mass"].unique()) == [1.0, 2.0, 3.0]


def test_trimming_a_complete_grid_drops_nothing():
    """
    Given a grid that is already a full rectangle,
    When it is trimmed,
    Then nothing is dropped and every row survives.
    """
    # Arrange
    df = _track_table([(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 0.5)])

    # Act
    trimmed, dropped = mist_grid.trim_to_complete_grid(df)

    # Assert
    assert dropped == {"dropped_mass": [], "dropped_initfeh": []}
    assert len(trimmed) == len(df)


def test_the_cut_axis_is_selectable():
    """
    Given a grid whose hole could be cleared along either axis,
    When each cut is requested,
    Then that axis is the one cut, and 'greedy' picks the cheaper one.
    """
    # Arrange: one hole at (2.0, -1.0); cutting feh drops 1 value, cutting
    # mass drops 1 value too, so greedy's tie-break (feh) is exercised.
    pairs = [(1.0, -1.0), (1.0, 0.0), (2.0, 0.0)]
    df = _track_table(pairs)

    # Act
    _, by_feh = mist_grid.trim_to_complete_grid(df, cut="feh")
    _, by_mass = mist_grid.trim_to_complete_grid(df, cut="mass")
    _, by_greedy = mist_grid.trim_to_complete_grid(df, cut="greedy")

    # Assert
    assert by_feh == {"dropped_mass": [], "dropped_initfeh": [-1.0]}
    assert by_mass == {"dropped_mass": [2.0], "dropped_initfeh": []}
    assert by_greedy == by_feh


def test_an_unknown_cut_option_raises():
    """
    Given a cut option that is not one of the three,
    When trimming runs,
    Then it raises naming the valid options.
    """
    # Arrange
    df = _track_table([(1.0, -1.0), (1.0, 0.0), (2.0, 0.0)])

    # Act / Assert
    with pytest.raises(ValueError, match="Must be one of 'greedy', 'feh'"):
        mist_grid.trim_to_complete_grid(df, cut="sideways")


# ----------------------------------------------------------------------
# Filenames
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,alpha,vvcrit,expected",
    [
        ("MISTv2.5", 0.0, 0.0, "afe_p0_vvcrit0.0.grid.parquet"),
        ("MISTv2.5", 0.4, 0.4, "afe_p4_vvcrit0.4.grid.parquet"),
        ("MISTv2.5", -0.2, 0.0, "afe_m2_vvcrit0.0.grid.parquet"),
        ("MISTv1.2", 0.0, 0.0, "afe_p0.0_vvcrit0.0.grid.parquet"),
        ("MISTv1.2", -0.2, 0.4, "afe_m0.2_vvcrit0.4.grid.parquet"),
    ],
)
def test_the_grid_filename_matches_the_packaged_spelling(
    model, alpha, vvcrit, expected
):
    """
    Given a (model, alpha, vvcrit) slice,
    When its path is built,
    Then the filename carries the 'afe_' prefix and the release's own
    [alpha/Fe] token spelling.
    """
    # Act
    path = mist_grid.grid_path(model=model, alpha=alpha, vvcrit=vvcrit)

    # Assert
    assert path.name == expected
    assert path.parent.name == "EEPs"
    assert path.parent.parent.name == model


# ----------------------------------------------------------------------
# Locating the parquet: the published grid is fetched, never assumed
# ----------------------------------------------------------------------


def test_a_missing_default_grid_is_fetched_from_zenodo(monkeypatch, tmp_path):
    """
    Given the ~128 MB grid is git-ignored and not shipped in the package,
    When a fit asks for the default (MISTv2.5, 0.0, 0.0) and it is not on
    disk,
    Then the published copy is fetched through models/MIST/eep_grid -- the
    one reader of the published grids, which verifies size and md5 and
    caches in place -- rather than raising at the user.
    """
    # Arrange: a default root that happens to be empty, so the "not present"
    # branch is the one taken without touching the developer's real grid.
    from exozippy.models.MIST import eep_grid

    fetched = tmp_path / "afe_p0_vvcrit0.0.grid.parquet"
    fetched.write_bytes(b"")
    calls = []

    def fake_ensure(alpha=0.0, vvcrit=0.0):
        calls.append((alpha, vvcrit))
        return fetched

    monkeypatch.setattr(eep_grid, "ensure_eep_grid", fake_ensure)
    monkeypatch.setattr(mist_grid, "DEFAULT_MODEL_ROOT", tmp_path / "models")

    # Act
    path = mist_grid._resolve_grid_file(
        "MISTv2.5", 0.0, 0.0, tmp_path / "models"
    )

    # Assert
    assert calls == [(0.0, 0.0)]
    assert path == fetched


def test_an_existing_grid_is_not_refetched(monkeypatch, tmp_path):
    """
    Given the grid is already on disk,
    When it is resolved,
    Then nothing is downloaded -- the fetch is first-use only.
    """
    # Arrange
    from exozippy.models.MIST import eep_grid

    root = tmp_path / "models"
    here = root / "MIST" / "MISTv2.5" / "EEPs"
    here.mkdir(parents=True)
    (here / "afe_p0_vvcrit0.0.grid.parquet").write_bytes(b"")

    def explode(**kwargs):  # pragma: no cover - must not run
        raise AssertionError("should not fetch an existing grid")

    monkeypatch.setattr(eep_grid, "ensure_eep_grid", explode)
    monkeypatch.setattr(mist_grid, "DEFAULT_MODEL_ROOT", root)

    # Act
    path = mist_grid._resolve_grid_file("MISTv2.5", 0.0, 0.0, root)

    # Assert
    assert path.name == "afe_p0_vvcrit0.0.grid.parquet"


def test_an_explicit_model_root_is_never_auto_populated(
    monkeypatch, tmp_path
):
    """
    Given a user pointing `model_root:` at a tree of their own,
    When the grid is not there,
    Then it raises rather than quietly downloading into the packaged tree --
    an explicit root is a statement about which grid to use.
    """
    # Arrange
    from exozippy.models.MIST import eep_grid

    def explode(**kwargs):  # pragma: no cover - must not run
        raise AssertionError("should not fetch for an explicit model_root")

    monkeypatch.setattr(eep_grid, "ensure_eep_grid", explode)

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="never auto-populated"):
        mist_grid._resolve_grid_file(
            "MISTv2.5", 0.0, 0.0, tmp_path / "elsewhere"
        )


def test_an_unpublished_model_release_raises(monkeypatch, tmp_path):
    """
    Given only MISTv2.5 is published on Zenodo,
    When a MISTv1.2 grid is requested and is not on disk,
    Then it raises pointing at the local build workflow.
    """
    # Arrange
    monkeypatch.setattr(mist_grid, "DEFAULT_MODEL_ROOT", tmp_path / "models")

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="only 'MISTv2.5' is published"):
        mist_grid._resolve_grid_file(
            "MISTv1.2", 0.0, 0.0, tmp_path / "models"
        )


def test_the_v25_filename_is_not_reimplemented_here():
    """
    Given models/MIST/eep_grid owns the published grid's filename (kept
    byte-identical to the authoring script's),
    When this module builds a MISTv2.5 path,
    Then it delegates rather than carrying a fourth copy of the rule.
    """
    # Arrange
    from exozippy.models.MIST.eep_grid import eep_grid_filename

    # Act / Assert
    for alpha, vvcrit in ((0.0, 0.0), (0.4, 0.4), (-0.2, 0.0)):
        assert mist_grid.grid_path(
            alpha=alpha, vvcrit=vvcrit
        ).name == eep_grid_filename(alpha=alpha, vvcrit=vvcrit)


# ----------------------------------------------------------------------
# Sentinels and grid health
# ----------------------------------------------------------------------


def test_the_hydrogen_exhausted_feh_survives_the_load_unchanged():
    """
    Given feh_mist == 30.0 is REAL DATA -- the value the grid generator
    substitutes when the surface hydrogen fraction underflows, so
    log10(Z/X) does not divide by zero, meaning "no hydrogen left in this
    photosphere" and occurring only in the highest-mass stars,
    When the grid is assembled,
    Then it is carried through exactly, not filtered, clamped or held
    across.

    This is a regression guard with a specific bug in mind: an earlier
    version of mist_grid forward-filled the last "normal" feh_mist across
    these rows on the theory that 30.0 was an error sentinel.  That
    fabricated a solar-metallicity prediction for a hydrogen-exhausted
    photosphere -- the opposite of what the model says.  A sampled star at
    [Fe/H] ~ 0 SHOULD earn an enormous penalty against such a track point.
    """
    # Arrange: a track whose tail is hydrogen-exhausted and flagged.
    df = _track_table(
        [(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 0.5)], eeps=(1, 2, 3)
    )
    tail = df["EEP"] == 3
    df.loc[tail, "feh_mist"] = mist_grid.FEH_HYDROGEN_EXHAUSTED
    df.loc[tail, "here_be_dragons"] = 1.0

    # Act
    grid = mist_grid._assemble_grid(df)

    # Assert
    feh = grid["values"][..., mist_grid.OUTPUT_INDEX["feh_mist"]]
    assert (feh[:, :, 2] == mist_grid.FEH_HYDROGEN_EXHAUSTED).all()
    dragons = grid["values"][..., mist_grid.OUTPUT_INDEX["here_be_dragons"]]
    assert (dragons[:, :, 2] == 1.0).all()


def _unresolved_track(ages, dragons):
    """One track with the given ages (yr) and generator dragon flags."""
    df = _track_table([(1.0, 0.0)], eeps=tuple(range(1, len(ages) + 1)))
    df = df.sort_values(["mass", "initfeh", "EEP"]).reset_index(drop=True)
    df["age_mist"] = list(ages)
    df["here_be_dragons"] = list(dragons)
    return df


def test_an_unresolved_age_run_is_found_by_its_one_year_step():
    """
    Given the grid generator adds exactly 1 yr to each successive duplicate
    age, so a run the stellar model could not resolve reaches us as steps of
    exactly 1.0 yr rather than as duplicates,
    When a track is scanned,
    Then the run's first step is what is found.
    """
    # Arrange: two ordinary steps, then the bumped run.
    d_age = np.array([np.nan, 1000.0, 500.0, 1.0, 1.0, 1.0])

    # Act / Assert
    assert mist_grid._first_unresolved_age_index(d_age) == 3


def test_genuinely_fast_evolution_is_not_mistaken_for_a_bumped_run():
    """
    Given real pre-main-sequence steps are dense in (0, 1) yr -- a 110
    solMass star's EEP 1 is ~4e-3 yr old -- while the bumping signature is
    exactly 1.0,
    When such a track is scanned,
    Then nothing is flagged.

    This is why the test is `== 1.0` and not `<= 1.0`: the loose version
    would flag the ordinary PMS of every massive track from EEP 1.
    """
    # Arrange
    d_age = np.array([np.nan, 3.4e-4, 4.4e-4, 0.9, 0.999, 2.0])

    # Act / Assert
    assert mist_grid._first_unresolved_age_index(d_age) == -1


def test_a_track_with_no_unresolved_step_is_left_alone():
    """
    Given a well-behaved track,
    When the flagging pass runs,
    Then nothing is raised.
    """
    # Arrange
    df = _unresolved_track([100.0, 200.0, 300.0, 400.0], [0, 0, 0, 0])

    # Act
    out, raised = mist_grid._flag_from_first_unresolved_age(df)

    # Assert
    assert raised == 0
    assert out["here_be_dragons"].tolist() == [0, 0, 0, 0]


def test_the_flag_is_extended_across_the_generators_zero_window():
    """
    Given the generator flags the duplicated rows but SKIPS the unique ages
    interleaved among them -- so a track drops back to here_be_dragons == 0
    across exactly the rows whose ages sit below the bumped ones and whose
    dEEP_dage is worst (track (90, -2.25) does this for EEP 618-631),
    When the flagging pass runs,
    Then the flag is a plain 1, 2, 3, ... count from the first unresolved
    step to the end of the track, with no gap.
    """
    # Arrange: rows 2-4 are the bumped run, row 5 is the unique age that
    # follows (and steps backwards), which the generator left unflagged.
    df = _unresolved_track(
        ages=[100.0, 200.0, 201.0, 202.0, 203.0, 150.0, 204.0],
        dragons=[0, 0, 1, 2, 3, 0, 4],
    )

    # Act
    out, raised = mist_grid._flag_from_first_unresolved_age(df)

    # Assert
    assert out["here_be_dragons"].tolist() == [0, 0, 1, 2, 3, 4, 5]
    assert raised == 2  # the zero-window row, and the resumed count
    flags = out["here_be_dragons"].to_numpy()
    tail = flags[np.flatnonzero(flags)[0]:]
    assert np.all(np.diff(tail) >= 0)


def test_the_flag_is_only_ever_raised_never_lowered():
    """
    Given the fix belongs in the grid generator and this pass is a stopgap,
    When it runs against a grid that already flags the whole tail,
    Then it changes nothing -- so it becomes a no-op the day the generator
    does this itself, rather than fighting it.
    """
    # Arrange: already correct, and with counts HIGHER than a plain 1..n.
    df = _unresolved_track(
        ages=[100.0, 200.0, 201.0, 202.0, 203.0],
        dragons=[0, 0, 10, 20, 30],
    )

    # Act
    out, raised = mist_grid._flag_from_first_unresolved_age(df)

    # Assert
    assert raised == 0
    assert out["here_be_dragons"].tolist() == [0, 0, 10, 20, 30]


def test_an_age_decrease_outside_a_flagged_tail_is_reported():
    """
    Given every decrease this module understands is a consequence of the +1
    yr bumping and therefore sits inside a flagged tail,
    When one appears anywhere else,
    Then it is reported -- that would mean something unexplained.
    """
    # Arrange
    df = _unresolved_track([100.0, 500.0, 200.0, 600.0], [0, 0, 0, 0])

    # Act
    bad = mist_grid._unflagged_nonmonotone_tracks(df)

    # Assert
    assert bad == [(1.0, 0.0)]


def test_a_flagged_decrease_is_not_reported():
    """
    Given the bumped-run decreases are now flagged,
    When the check runs after the flagging pass,
    Then nothing is reported.
    """
    # Arrange
    df = _unresolved_track(
        ages=[100.0, 200.0, 201.0, 202.0, 150.0], dragons=[0, 0, 1, 2, 0]
    )

    # Act
    flagged, _ = mist_grid._flag_from_first_unresolved_age(df)

    # Assert
    assert mist_grid._unflagged_nonmonotone_tracks(flagged) == []


# ----------------------------------------------------------------------
# Assembling the dense array
# ----------------------------------------------------------------------


def test_the_dense_array_is_indexed_by_mass_then_feh_then_eep():
    """
    Given a complete grid,
    When it is assembled,
    Then the axes are ascending and values[m, f, e] is that track point.
    """
    # Arrange
    df = _track_table([(1.0, 0.0), (1.0, 0.5), (10.0, 0.0), (10.0, 0.5)])

    # Act
    grid = mist_grid._assemble_grid(df)

    # Assert
    assert grid["values"].shape == (2, 2, 3, len(mist_grid.OUTPUT_COLUMNS))
    np.testing.assert_allclose(grid["logmass_pts"], [0.0, 1.0])
    np.testing.assert_allclose(grid["initfeh_pts"], [0.0, 0.5])
    np.testing.assert_allclose(grid["eep_pts"], [1.0, 2.0, 3.0])
    feh = grid["values"][:, :, :, mist_grid.OUTPUT_INDEX["feh_mist"]]
    np.testing.assert_allclose(feh[0, 1, :], [0.5, 0.51, 0.52])


def test_ages_are_converted_to_gyr():
    """
    Given a table whose age_mist is in years,
    When the grid is assembled,
    Then the stacked age output is in Gyr, matching star.age's unit.
    """
    # Arrange
    df = _track_table([(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 0.5)])

    # Act
    grid = mist_grid._assemble_grid(df)

    # Assert
    ages = grid["values"][0, 0, :, mist_grid.OUTPUT_INDEX["age_mist_gyr"]]
    np.testing.assert_allclose(ages, [1.0, 2.0, 3.0])


def test_eeps_past_the_supported_maximum_are_dropped():
    """
    Given a regenerated grid running past MIST's EEP 807,
    When it is assembled,
    Then the extra points are truncated rather than silently extending the
    axis into the TPAGB regime this component does not model.
    """
    # Arrange
    eeps = (mist_grid.MAX_VALID_EEP - 1, mist_grid.MAX_VALID_EEP,
            mist_grid.MAX_VALID_EEP + 1)
    df = _track_table([(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 0.5)],
                      eeps=eeps)

    # Act
    grid = mist_grid._assemble_grid(df)

    # Assert
    assert grid["eep_pts"].max() == float(mist_grid.MAX_VALID_EEP)
    assert len(grid["eep_pts"]) == 2


def test_a_grid_with_a_short_track_raises():
    """
    Given one track with fewer EEPs than the rest,
    When the grid is assembled,
    Then the row-count guard fires rather than the reshape mis-indexing.
    """
    # Arrange
    a = _track_table([(1.0, 0.0), (1.0, 0.5), (2.0, 0.0)], eeps=(1, 2, 3))
    b = _track_table([(2.0, 0.5)], eeps=(1, 2, 4))
    df = pd.concat([a, b], ignore_index=True)

    # Act / Assert
    with pytest.raises(ValueError, match="not a complete .* rectangle"):
        mist_grid._assemble_grid(df)


def test_a_ragged_eep_array_raises_even_at_the_right_row_count():
    """
    Given a track whose EEP array differs from the others but has the same
    LENGTH (a duplicated EEP), so the row count check passes,
    When the grid is assembled,
    Then the per-track EEP comparison catches it rather than the reshape
    silently mis-indexing that track.
    """
    # Arrange
    good = _track_table(
        [(1.0, 0.0), (1.0, 0.5), (2.0, 0.0)], eeps=(1, 2, 3, 4)
    )
    ragged = _track_table([(2.0, 0.5)], eeps=(1, 2, 3, 3))
    df = pd.concat([good, ragged], ignore_index=True)

    # Act / Assert
    with pytest.raises(ValueError, match="EEP array differs"):
        mist_grid._assemble_grid(df)


# ----------------------------------------------------------------------
# The systematic floor
# ----------------------------------------------------------------------


def test_the_systematic_floor_matches_exofastv2():
    """
    Given massradius_mist.pro's percenterror expression,
    When the port is evaluated,
    Then it agrees -- and its two entry points (numpy from log10(mass),
    pytensor from mass) agree with each other.
    """
    # Arrange
    for mstar in (0.3, 1.0, 2.5):
        expected = (
            0.03 - 0.025 * np.log10(mstar) + 0.045 * np.log10(mstar) ** 2
        )

        # Act
        from_log = physics.percent_error_from_logmass(np.log10(mstar))
        from_mass = physics.calc_mist_percent_error(
            pt.as_tensor_variable(np.float64(mstar))
        ).eval()

        # Assert
        assert from_log == pytest.approx(expected)
        assert float(from_mass) == pytest.approx(expected)


def test_the_floor_bottoms_out_near_two_solar_masses_not_at_the_sun():
    """
    Given the floor is a parabola in log10(mstar) whose vertex is at
    -_B/(2*_C) = 0.2778 dex = 1.90 solMass, NOT at the Sun,
    When it is evaluated across the range,
    Then it is 3 per cent at 1 solMass, minimized at ~1.9, and rises steeply
    only toward LOW mass.

    Written this way deliberately: the obvious "3% at the Sun and rising in
    both directions" reading is wrong, and 3 solMass sits BELOW the solar
    value.  See physics.calc_mist_percent_error's docstring.
    """
    # Act
    at_one = physics.percent_error_from_logmass(0.0)
    light = physics.percent_error_from_logmass(np.log10(0.3))
    vertex = physics.percent_error_from_logmass(0.025 / (2 * 0.045))
    heavy = physics.percent_error_from_logmass(np.log10(3.0))

    # Assert
    assert at_one == pytest.approx(0.03)
    assert light == pytest.approx(0.05538, rel=1e-3)
    assert vertex == pytest.approx(0.02653, rel=1e-3)
    assert vertex < heavy < at_one < light


# ----------------------------------------------------------------------
# The EEP -> age Jacobian
# ----------------------------------------------------------------------


def _jacobian_logp(deep_dage):
    """The potential exactly as build_likelihood forms it."""
    x = pt.as_tensor_variable(np.asarray(deep_dage, dtype=float))
    return float(
        (
            -pt.sum(pt.log(pt.clip(pt.abs(x), DEEP_DAGE_MIN, DEEP_DAGE_MAX)))
        ).eval()
    )


def test_slow_evolution_is_favored_over_fast():
    """
    Given two EEPs, one on a slowly evolving phase (small dEEP/dage),
    When the Jacobian potential is evaluated at each,
    Then the slow one scores higher -- which is what makes a flat prior in
    EEP a flat prior in age (most of a star's life is spent slowly).
    """
    # Act
    slow = _jacobian_logp([1e-10])
    fast = _jacobian_logp([1e-6])

    # Assert
    assert slow > fast
    assert slow - fast == pytest.approx(np.log(1e4))


def test_the_jacobian_reward_is_bounded_below_and_above():
    """
    Given a defective dEEP_dage -- near zero (an interpolated sign flip
    between the +/-2**32 entries) or absurdly large,
    When the potential is evaluated,
    Then the clip caps it, so neither is an unbounded attractor or wall.

    The floor this replaced was 1e-30, i.e. +69 nats of reward for landing
    on a table defect.
    """
    # Act
    at_zero = _jacobian_logp([0.0])
    absurdly_small = _jacobian_logp([1e-40])
    absurdly_large = _jacobian_logp([2.0**32])

    # Assert
    assert at_zero == pytest.approx(-np.log(DEEP_DAGE_MIN))
    assert absurdly_small == at_zero
    assert absurdly_large == pytest.approx(-np.log(DEEP_DAGE_MAX))
    assert at_zero < -np.log(1e-30)


def test_the_jacobian_sign_matches_exofastv2_ageweight():
    """
    Given EXOFASTv2's `chi2 -= 2*alog(ageweight)` with ageweight = da/dEEP,
    When our potential is compared to +log|da/de|,
    Then they agree up to the additive constant from the yr -> Gyr rescale.
    """
    # Arrange
    de_da = np.array([1e-9, 4e-9, 2.5e-10])

    # Act
    ours = _jacobian_logp(de_da)
    exofast = float(np.sum(np.log(1.0 / de_da)))

    # Assert
    assert ours == pytest.approx(exofast)


# ----------------------------------------------------------------------
# The component's parameter layout
# ----------------------------------------------------------------------


def test_the_component_declares_no_parameters_of_its_own():
    """
    Given that star.initfeh/star.eep/star.age already exist and are
    materialized by Star's in_system("evolutionarymodel") branch,
    When this component registers,
    Then its manifest is empty -- a second copy would leave the star's
    materialized, free, and read by nothing.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)

    # Act
    comp.register_parameters(system=None)

    # Assert
    assert comp.manifest == {}


def test_the_component_ships_no_defaults_yaml():
    """
    Given the component owns no parameters,
    When its directory is listed,
    Then there is no defaults.yaml for ConfigManager to load a duplicate
    initfeh/eep block from.
    """
    # Arrange
    import exozippy.components.evolutionarymodel as pkg
    from pathlib import Path

    # Act
    here = Path(pkg.__file__).parent

    # Assert
    assert not (here / "defaults.yaml").exists()


def test_constrain_parsing_is_the_shared_mixin_implementation():
    """
    Given this component constrains feh/radius/teff/age, not mass/radius,
    When `constrain:` is parsed,
    Then it uses the mixin's one copy via the `constrainable` hook rather
    than a near-identical override.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)

    # Assert
    assert EvolutionaryModel._parse_constrain is StellarRelation._parse_constrain
    assert comp.constrainable == CONSTRAINABLE
    assert comp._parse_constrain("A", None) == set(CONSTRAINABLE)
    assert comp._parse_constrain("A", "teff") == {"teff"}


@pytest.mark.parametrize("bad", [["mass"], ["logg"], ["eep"]])
def test_quantities_the_tracks_do_not_constrain_are_rejected(bad):
    """
    Given a `constrain:` naming something this component does not tie,
    When it is parsed,
    Then it raises listing what is valid.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)

    # Act / Assert
    with pytest.raises(ValueError, match="unknown 'constrain:' entries"):
        comp._parse_constrain("A", bad)


# ----------------------------------------------------------------------
# The per-star model switches
# ----------------------------------------------------------------------


class _FakeStar:
    def __init__(self, names, mist, parsec):
        self.names = list(names)
        self.mist = list(mist)
        self.parsec = list(parsec)


class _FakeSystem:
    def __init__(self, star):
        self.star = star


def test_a_star_that_opted_out_of_every_model_raises():
    """
    Given a star with `mist: False` and no `parsec:`,
    When an evolutionarymodel block names it,
    Then it raises -- the config says one thing and would do another.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)
    system = _FakeSystem(_FakeStar(["A"], [False], [False]))

    # Act / Assert
    with pytest.raises(ValueError, match="opted out of every evolutionary"):
        comp._check_star_opted_in(system, "A", 0)


def test_a_parsec_only_star_raises_rather_than_getting_mist():
    """
    Given a star asking for PARSEC alone,
    When an evolutionarymodel block names it,
    Then it raises, because only MIST is implemented and it would otherwise
    be handed MIST tracks silently.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)
    system = _FakeSystem(_FakeStar(["A"], [False], [True]))

    # Act / Assert
    with pytest.raises(ValueError, match="only the MIST tracks"):
        comp._check_star_opted_in(system, "A", 0)


def test_a_mist_star_is_accepted():
    """
    Given the default `mist: True`,
    When the switch is checked,
    Then nothing is raised.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)
    system = _FakeSystem(_FakeStar(["A"], [True], [False]))

    # Act / Assert
    comp._check_star_opted_in(system, "A", 0)


# ----------------------------------------------------------------------
# Bounds and pins pushed at other components' parameters
# ----------------------------------------------------------------------


class _RecordingConfigManager:
    def __init__(self):
        self.overrides = {}
        self.hints = {}

    def add_override(self, path, **fields):
        self.overrides.setdefault(path, {}).update(fields)

    def add_hint(self, path, value, rank=None):
        self.hints[path] = value


def _grid_stub():
    return {
        "logmass_pts": np.array([-1.0, 0.0, 1.0]),
        "initfeh_pts": np.array([-2.5, 0.0, 0.5]),
        "eep_pts": np.array([1.0, 400.0, 807.0]),
    }


def test_the_grid_extents_become_bounds_on_the_stars_own_parameters():
    """
    Given an interpolator that extrapolates meaninglessly past its axes,
    When the component loads its grid,
    Then it bounds star.logmass/initfeh/eep to the grid extent through
    add_override -- NOT by writing into config_manager.user_params, which
    would report the bound as the user's own.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)
    comp.config_manager = _RecordingConfigManager()

    # Act
    comp._inject_grid_bounds("A", _grid_stub())

    # Assert
    assert comp.config_manager.overrides == {
        "star.A.logmass": {"lower": -1.0, "upper": 1.0},
        "star.A.initfeh": {"lower": -2.5, "upper": 0.5},
        "star.A.eep": {"lower": 1.0, "upper": 807.0},
    }


def test_stars_no_instance_names_get_their_track_parameters_pinned(caplog):
    """
    Given Star materializes initfeh/eep/age for EVERY star as soon as an
    evolutionarymodel block exists (its per-star `mask` is the unconsumed
    manifest field),
    When the component loads,
    Then the stars it does not model have all three pinned, so they are not
    free dimensions no likelihood term reads.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)
    comp.config_manager = _RecordingConfigManager()
    comp.star_indices = [0]
    system = _FakeSystem(_FakeStar(["A", "B"], [True, True], [False, False]))

    # Act
    with caplog.at_level(logging.INFO):
        comp._pin_unmodeled_stars(system)

    # Assert
    assert comp.config_manager.overrides == {
        "star.B.initfeh": {"sigma": 0.0},
        "star.B.eep": {"sigma": 0.0},
        "star.B.age": {"sigma": 0.0},
    }
    assert "no evolutionarymodel block names it" in caplog.text


# ----------------------------------------------------------------------
# End to end, against a synthetic grid
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    """A tiny MIST grid; the shipped one is ~130 MB and gitignored."""
    from conftest import write_synthetic_mist_grid

    return write_synthetic_mist_grid(tmp_path_factory.mktemp("mist_models"))


@pytest.fixture(scope="module")
def built(model_root):
    """A one-star system with the component, prepared and built."""
    from exozippy.system import System

    config = {
        "sampler": {"draws": 10},
        "star": [{"name": "A"}],
        "evolutionarymodel": [{"star": "A", "model_root": model_root}],
    }
    system = System(config, {})
    system.prepare()
    model = system.build_model()
    return system, model


def test_the_model_builds_and_adds_one_potential_per_term(built):
    """
    Given a star tied to the tracks with the default `constrain:`,
    When the model is built,
    Then the four penalties, the EEP Jacobian and the dragon penalty are all
    present and finite at the start point.
    """
    # Arrange
    system, model = built
    expected = {f"evolutionarymodel.{q}_prior" for q in CONSTRAINABLE} | {
        "evolutionarymodel.eep_age_jacobian",
        "evolutionarymodel.dragon_penalty",
    }

    # Act
    names = {p.name for p in model.potentials}

    # Assert
    assert expected <= names
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_the_track_coordinates_are_the_stars_own_parameters(built):
    """
    Given the component declares no parameters of its own,
    When the model is built,
    Then the sampled track coordinates are star.initfeh/star.eep and nothing
    is registered under this component's prefix -- no duplicate pair.
    """
    # Arrange
    system, model = built

    # Act
    raw = {v.name for v in model.free_RVs}

    # Assert
    assert {"star.initfeh_raw", "star.eep_raw"} <= raw
    assert not any(r.startswith("evolutionarymodel.") for r in raw)


def test_the_predictions_are_reported_as_deterministics(built):
    """
    Given a fit a user has to be able to inspect,
    When the model is built,
    Then each MIST prediction is a named Deterministic.
    """
    # Arrange
    system, model = built

    # Act
    names = {d.name for d in model.deterministics}

    # Assert
    assert {
        "evolutionarymodel.feh_pred",
        "evolutionarymodel.radius_pred",
        "evolutionarymodel.teff_pred",
        "evolutionarymodel.age_pred",
        "evolutionarymodel.here_be_dragons",
    } <= names


def test_the_added_priors_are_declared_to_the_tables(built):
    """
    Given a component that adds pm.Potentials at stage 6,
    When the Prior column is rendered,
    Then it names the MIST penalty and the EEP Jacobian rather than the
    "Uniform" those terms replace -- the add_prior_contribution contract.
    """
    # Arrange
    system, _ = built

    # Act
    teff_prior = system.star.teff.get_prior_str(index=0, latex=False)
    eep_prior = system.star.eep.get_prior_str(index=0, latex=False)

    # Assert
    assert "MIST" in teff_prior
    assert "dAge/dEEP" in eep_prior
    assert "Uniform" not in eep_prior


def test_the_modeling_draft_cites_the_tracks(built):
    """
    Given the declare-at-the-implementation-site prose rule,
    When the model is built,
    Then the MIST sentences are collected with their citation keys.
    """
    # Arrange
    from exozippy.outputs.prose import get_collector

    system, _ = built
    prose = get_collector(system)

    # Act
    text = " ".join(s.text for s in prose.sentences())

    # Assert
    assert "MIST evolutionary" in text
    assert "uniform prior on stellar age" in text
    assert {"Dotter:2016", "Choi:2016"} <= set(prose.cite_keys())


def test_a_block_naming_no_star_warns_and_builds(model_root, caplog):
    """
    Given an `evolutionarymodel:` block with no entries,
    When the system is prepared,
    Then it warns rather than dying inside pt.stack, and the track
    parameters the block's mere presence materialized are pinned.
    """
    # Arrange
    from exozippy.system import System

    config = {
        "sampler": {"draws": 10},
        "star": [{"name": "A"}],
        "evolutionarymodel": [],
    }

    # Act
    with caplog.at_level(logging.WARNING):
        system = System(config, {})
        system.prepare()
        system.build_model()

    # Assert
    assert "names no star" in caplog.text
    assert not system.star.eep.is_sampled[0]


# ----------------------------------------------------------------------
# Interpolating a track, rather than snapping to the nearest one
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_grid():
    """A 2x2x3 grid whose four corner tracks are all distinct."""
    df = _track_table([(1.0, 0.0), (1.0, 0.5), (10.0, 0.0), (10.0, 0.5)])
    return mist_grid._assemble_grid(df)


def test_interpolating_on_a_grid_node_reproduces_that_track(toy_grid):
    """
    Given a query sitting exactly on a tabulated (logmass, initfeh) node,
    When the track is interpolated,
    Then it is that node's track, bit for bit.
    """
    # Act
    track = mist_grid.interpolate_track(toy_grid, 0.0, 0.5)

    # Assert
    np.testing.assert_array_equal(track, toy_grid["values"][0, 1])


def test_interpolating_at_a_cell_midpoint_blends_all_four_corners(toy_grid):
    """
    Given a query at the centre of a grid cell,
    When the track is interpolated,
    Then it is the mean of the four surrounding tracks -- the same bilinear
    blend the likelihood's own interpolator applies, so the seed search and
    the fit see one surface.
    """
    # Arrange
    v = toy_grid["values"]
    mid_logmass = 0.5 * (
        toy_grid["logmass_pts"][0] + toy_grid["logmass_pts"][1]
    )
    mid_feh = 0.5 * (toy_grid["initfeh_pts"][0] + toy_grid["initfeh_pts"][1])

    # Act
    track = mist_grid.interpolate_track(toy_grid, mid_logmass, mid_feh)

    # Assert
    np.testing.assert_allclose(
        track, 0.25 * (v[0, 0] + v[1, 0] + v[0, 1] + v[1, 1])
    )


def test_a_query_outside_the_grid_clamps_rather_than_extrapolating(toy_grid):
    """
    Given a query beyond an axis edge,
    When the track is interpolated,
    Then it is pinned to the edge track.

    Deliberately different from the likelihood's interpolator, which
    EXTRAPOLATES (and which the component's grid bounds exist to keep it
    from having to do): a seed wants the nearest point the models actually
    cover, and an extrapolated start value means nothing.
    """
    # Act
    below = mist_grid.interpolate_track(toy_grid, -99.0, 0.0)
    above = mist_grid.interpolate_track(toy_grid, +99.0, 0.0)

    # Assert
    np.testing.assert_array_equal(below, toy_grid["values"][0, 0])
    np.testing.assert_array_equal(above, toy_grid["values"][-1, 0])


def test_the_eep_seed_reads_an_interpolated_track_not_the_nearest_node(
    monkeypatch, toy_grid
):
    """
    Given the star sits between tabulated tracks,
    When the EEP seed is searched,
    Then the objective is evaluated on the INTERPOLATED track.

    Neighbouring tracks differ everywhere, most of all in age, so a
    nearest-node search can seed a fit on a track up to half a cell away in
    both axes.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)
    comp.config_manager = _RecordingConfigManager()
    comp.names = ["A"]
    comp.constrain = [{"teff"}]
    for attr in ("feh_floor", "radius_floor", "teff_floor", "age_floor"):
        setattr(comp, attr, [None])
    comp.dragon_penalty_weight = [1.0]
    comp._grids = [toy_grid]

    starts = {"logmass": 0.5, "initfeh": 0.25, "teff": 5150.0}
    comp.config_manager.resolve = lambda c, p, element=None: {
        "initval": starts.get(p, 1.0)
    }

    seen = {}
    real = mist_grid.interpolate_track

    def spy(grid, logmass, initfeh):
        seen["coords"] = (logmass, initfeh)
        return real(grid, logmass, initfeh)

    monkeypatch.setattr(mist_grid, "interpolate_track", spy)

    # Act
    comp._seed_eep_hint(0, "A", 0, toy_grid)

    # Assert: queried at the star's own coordinates, not a snapped node
    assert seen["coords"] == (0.5, 0.25)
    assert "star.A.eep" in comp.config_manager.hints


# ----------------------------------------------------------------------
# The Kiel diagram
# ----------------------------------------------------------------------


def _kiel_spec(system, component):
    point = {p.label: p.initval for p in system.plot_params}
    specs = component.plot_data(system, point)
    assert len(specs) == 1
    return specs[0]


def test_without_a_point_there_is_nothing_to_draw(built):
    """
    Given this component contributes only potentials and has no
    observations of its own,
    When plot_data is asked for a data-only preview,
    Then it returns nothing -- every curve it draws is a model quantity that
    needs a parameter point to exist.
    """
    # Arrange
    system, _ = built

    # Act / Assert
    assert (
        system.active_components["evolutionarymodel"].plot_data(system, None)
        == []
    )


def test_the_kiel_diagram_draws_track_prediction_and_fit(built):
    """
    Given a star tied to the tracks,
    When the Kiel diagram is built at a point,
    Then it carries three curves per star: the full EEP track, the MIST
    prediction at the fitted EEP, and the fitted point itself.
    """
    # Arrange
    system, _ = built
    comp = system.active_components["evolutionarymodel"]

    # Act
    spec = _kiel_spec(system, comp)

    # Assert
    assert [t.name for t in spec.traces] == [
        "A MIST track",
        "A MIST point",
        "A fit +/- systematic",
    ]
    assert [t.kind for t in spec.traces] == ["line", "scatter", "scatter"]
    assert np.atleast_1d(spec.traces[0].x).size > 1  # a curve, not a point
    for trace in spec.traces[1:]:
        assert np.atleast_1d(trace.x).size == 1


def test_the_kiel_axes_follow_the_convention(built):
    """
    Given the Kiel-diagram convention -- Teff decreasing rightward, surface
    gravity increasing downward, so dwarfs sit low as in an observational HR
    diagram,
    When the spec is built,
    Then both axes are reversed and it names its own PDF.
    """
    # Arrange
    system, _ = built

    # Act
    spec = _kiel_spec(system, system.active_components["evolutionarymodel"])

    # Assert
    assert spec.meta["x_inverted"] and spec.meta["y_inverted"]
    assert spec.meta["file_tag"] == "kiel"
    assert "logg" in spec.ylabel or r"\log g" in spec.ylabel


def test_the_fitted_point_carries_the_systematic_floor_as_error_bars(built):
    """
    Given the systematic floor is what lets the fit sit away from the track,
    When the fitted point is drawn,
    Then its error bars ARE that floor: the Teff bar is the fractional floor
    times the MIST prediction, and the logg bar is the radius floor
    propagated through logg = C + logmass - 2*log10(R), i.e. 2*f_R/ln(10).

    logg is neither a grid column nor a constrained quantity, so this
    propagation is the only thing that can put a meaningful bar on it -- and
    the mass contributes nothing, carrying no floor of its own.
    """
    # Arrange
    system, _ = built
    comp = system.active_components["evolutionarymodel"]
    spec = _kiel_spec(system, comp)
    track_trace, mist_trace, fit_trace = spec.traces

    logmass = float(np.atleast_1d(system.star.logmass.initval)[0])
    floor = physics.percent_error_from_logmass(logmass)
    teff_pred = float(np.atleast_1d(mist_trace.x)[0])

    # Assert
    assert float(np.atleast_1d(fit_trace.xerr)[0]) == pytest.approx(
        floor * teff_pred, rel=1e-6
    )
    assert float(np.atleast_1d(fit_trace.yerr)[0]) == pytest.approx(
        2.0 * floor / np.log(10.0), rel=1e-6
    )


def test_the_kiel_diagram_declares_what_moves_it(built):
    """
    Given an empty param_deps makes the GUI's live mode freeze a chart,
    When the spec is built,
    Then it names the sampled parameters its curves actually depend on.
    """
    # Arrange
    system, _ = built

    # Act
    spec = _kiel_spec(system, system.active_components["evolutionarymodel"])

    # Assert
    assert {"star.logmass", "star.initfeh", "star.eep"} <= set(
        spec.param_deps
    )


def test_the_track_stops_where_the_models_stop_being_trustworthy():
    """
    Given the tail of a track is flagged here_be_dragons,
    When the drawn curve is built,
    Then the flagged rows are cut -- the fit may still go there (the penalty
    is smooth, not a wall), but the chart should not draw a curve through
    the hydrogen-exhausted regime as though it were reliable.
    """
    # Arrange
    comp = EvolutionaryModel([{"star": "A"}], None)
    df = _track_table(
        [(1.0, 0.0), (1.0, 0.5), (10.0, 0.0), (10.0, 0.5)], eeps=(1, 2, 3, 4)
    )
    df.loc[df["EEP"] >= 3, "here_be_dragons"] = 1.0
    comp._grids = [mist_grid._assemble_grid(df)]

    # Act -- an EEP window wide enough to keep every synthetic row, so the
    # dragon cut is the only thing under test here.
    teff, logg = comp._track_curve(0, 0.0, 0.0, (0.0, 1.0e4))

    # Assert
    assert teff.size == 2  # EEP 1 and 2 only
    assert np.all(np.isfinite(logg))


# ----------------------------------------------------------------------
# The Kiel diagram's plotting windows, and its once-per-fit marks
# ----------------------------------------------------------------------


def _kiel_system(model_root, user_params=None):
    """A fresh one-star system, for tests that mutate plotting state.

    The `built` fixture is module-scoped, and `_reported_kiel` memoizes, so
    a test that attaches a posterior would leak into every later test.
    """
    from exozippy.system import System

    system = System(
        {
            "sampler": {"draws": 10},
            "star": [{"name": "A"}],
            "evolutionarymodel": [{"star": "A", "model_root": model_root}],
        },
        dict(user_params or {}),
    )
    system.prepare()
    system.build_model()
    return system, system.active_components["evolutionarymodel"]


def test_a_value_comfortably_inside_the_window_leaves_it_alone():
    """
    Given a window and a value well inside it,
    When the window is extended,
    Then nothing moves -- the nominal window is the answer in the ordinary
    case, and widening it for a star it already contains would throw away
    the framing it exists to provide.
    """
    # Act
    lo, hi = EvolutionaryModel._extend_window((3.0, 5.0), [4.0], 0.2)

    # Assert
    assert (lo, hi) == (3.0, 5.0)


@pytest.mark.parametrize(
    "value, expected",
    [
        (3.1, (2.9, 5.0)),  # NEAR the lower edge (within the margin)
        (2.0, (1.8, 5.0)),  # fully OUTSIDE it
        (4.95, (3.0, 5.15)),  # near the upper edge
        (7.0, (3.0, 7.2)),  # fully outside it
    ],
)
def test_the_window_widens_for_a_value_near_or_past_an_edge(value, expected):
    """
    Given a star sitting near a window edge, or past it,
    When the window is extended,
    Then that edge moves out to `value -/+ margin` -- "very near the bound"
    and "outside the bound" are the same comparison once the margin is
    folded in, and a window that clips the star it describes is worse than
    no window at all.
    """
    # Act
    got = EvolutionaryModel._extend_window((3.0, 5.0), [value], 0.2)

    # Assert
    assert got == pytest.approx(expected)


def test_the_drawn_track_is_restricted_to_the_requested_eeps(model_root):
    """
    Given the chart is about the main sequence and the red giant branch,
    When the track is drawn over the nominal EEP window,
    Then the pre-main-sequence and post-RGB rows are cut.

    The synthetic grid tabulates EEPs (1, 300, 454, 605, 807), so [202, 630]
    keeps exactly the three middle ones.
    """
    # Arrange
    _system, comp = _kiel_system(model_root)

    # Act
    kept = comp._track_curve(0, 0.0, 0.0, KIEL_EEP_WINDOW)[0]
    everything = comp._track_curve(0, 0.0, 0.0, (0.0, 1.0e4))[0]

    # Assert
    assert kept.size == 3
    assert everything.size == 5


def test_the_drawn_track_widens_for_a_star_past_the_eep_window(model_root):
    """
    Given a star whose fitted EEP sits past the nominal window,
    When the Kiel diagram is built,
    Then the drawn arc is widened to reach it -- the window is presentation,
    not physics, and must never hide the star the chart describes.
    """
    # Arrange
    _base_system, base = _kiel_system(model_root)
    base_spec = _kiel_spec(_base_system, base)
    system, comp = _kiel_system(model_root, {"star.A.eep": 800.0})

    # Act
    spec = _kiel_spec(system, comp)

    # Assert -- EEP 807 is inside 800 + margin, and was outside 630
    assert np.atleast_1d(base_spec.traces[0].x).size == 3
    assert np.atleast_1d(spec.traces[0].x).size == 4


def test_the_logg_axis_is_the_nominal_window_for_a_dwarf(built):
    """
    Given the chart is meant to exclude giants by default,
    When a main-sequence star is drawn,
    Then the logg axis is exactly the nominal window.  It is stated ascending
    because plotrender applies `y_range` BEFORE `y_inverted`, so the drawn
    axis runs 5.0 at the bottom to 3.0 at the top.
    """
    # Arrange
    system, _ = built

    # Act
    spec = _kiel_spec(system, system.active_components["evolutionarymodel"])

    # Assert
    assert spec.meta["y_range"] == list(KIEL_LOGG_WINDOW)


def test_the_logg_axis_widens_for_a_star_outside_it(model_root):
    """
    Given a star whose fitted logg falls outside the nominal window,
    When the Kiel diagram is built,
    Then the axis is widened past it by the margin, rather than drawing a
    chart the star is not on.
    """
    # Arrange -- a 30 solRad star: logg = C + logmass - 2*log10(R) ~ 1.5
    system, comp = _kiel_system(model_root, {"star.A.radius": 30.0})

    # Act
    spec = _kiel_spec(system, comp)
    lo, hi = spec.meta["y_range"]
    fit_logg = float(np.atleast_1d(spec.traces[2].y)[0])
    margin = KIEL_WINDOW_MARGIN_FRAC * (
        KIEL_LOGG_WINDOW[1] - KIEL_LOGG_WINDOW[0]
    )

    # Assert
    assert fit_logg < KIEL_LOGG_WINDOW[0]
    assert lo <= fit_logg - margin
    assert hi == KIEL_LOGG_WINDOW[1]


def test_the_marks_are_drawn_once_however_many_draws_are_overlaid(built):
    """
    Given plotrender takes only role="model" traces from the non-reference
    draws,
    When the Kiel spec is built,
    Then both marks carry role="data" -- so 50 overlaid draws produce 50
    tracks but exactly one MIST point and one fitted point -- and both sit
    above the track spaghetti via an explicit zorder.
    """
    # Arrange
    system, _ = built

    # Act
    spec = _kiel_spec(system, system.active_components["evolutionarymodel"])
    track, mist_point, fit_point = spec.traces

    # Assert
    assert track.role == "model"
    assert mist_point.role == fit_point.role == "data"
    # _draw_model's default is 2; the marks must land above it.
    assert mist_point.style["zorder"] > 2
    assert fit_point.style["zorder"] > mist_point.style["zorder"]


def test_the_marks_sit_at_the_reported_medians_not_at_the_drawn_draw(
    model_root,
):
    """
    Given the marks are single numbers a reader compares against the table,
    When a posterior is attached and the diagram is drawn at some OTHER
    point,
    Then the marks stay at the posterior medians while the track follows the
    drawn point -- the tracks carry the uncertainty, the marks carry the
    reported answer.
    """
    # Arrange
    system, comp = _kiel_system(model_root)
    at_initval = {p.label: p.initval for p in system.plot_params}
    # A posterior pinned at each parameter's start value (user units, sample
    # dimension LAST -- the layout System.distribute_posterior produces).
    for param in system.plot_params:
        user = np.asarray(param.from_internal(param.initval), dtype=float)
        param.posterior = np.repeat(user[..., None], 7, axis=-1)
    assert comp._reported_kiel(system) is not None

    moved = dict(at_initval)
    moved["star.teff"] = np.atleast_1d(moved["star.teff"]) * 1.10

    # Act
    spec = comp.plot_data(system, moved)[0]
    _track, mist_point, fit_point = spec.traces
    reference = _kiel_spec(system, comp)

    # Assert -- the marks ignore the moved draw entirely
    assert float(np.atleast_1d(fit_point.x)[0]) == pytest.approx(
        float(np.atleast_1d(reference.traces[2].x)[0])
    )
    assert float(np.atleast_1d(fit_point.x)[0]) == pytest.approx(
        float(np.atleast_1d(at_initval["star.teff"])[0])
    )
    assert float(np.atleast_1d(mist_point.y)[0]) == pytest.approx(
        float(np.atleast_1d(reference.traces[1].y)[0])
    )


def test_without_a_posterior_the_marks_fall_back_to_the_drawn_point(built):
    """
    Given the pre-flight plot and the GUI's live-slider mode have no
    posterior to report,
    When the diagram is drawn,
    Then the marks come from the point they were handed -- which in both of
    those cases is the only point there is.
    """
    # Arrange
    system, _ = built
    comp = system.active_components["evolutionarymodel"]
    assert comp._reported_kiel(system) is None

    # Act
    kiel = np.atleast_2d(
        comp._compiled_kiel(*comp._point_to_plot_params(
            {p.label: p.initval for p in system.plot_params}, system
        ))
    )
    spec = _kiel_spec(system, comp)

    # Assert
    assert float(np.atleast_1d(spec.traces[2].x)[0]) == pytest.approx(
        float(kiel[0, KIEL_INDEX["teff_fit"]])
    )


def test_an_empty_set_of_values_leaves_the_axis_to_autoscale():
    """
    Given nothing finite is on the chart,
    When the padded range is computed,
    Then it is None -- the caller omits the range key rather than inventing
    an axis, which is the honest answer when there is nothing to scale to.
    """
    # Act / Assert
    assert EvolutionaryModel._padded_range([], 0.05) is None
    assert (
        EvolutionaryModel._padded_range([np.array([np.nan, np.inf])], 0.05)
        is None
    )


def test_a_single_point_still_gets_a_nonzero_axis_span():
    """
    Given one star, one draw, and a zero-width spread,
    When the padded range is computed,
    Then the pad comes from the value's own magnitude rather than leaving a
    zero-width axis the renderer cannot draw.
    """
    # Act
    lo, hi = EvolutionaryModel._padded_range([np.array([5000.0])], 0.05)

    # Assert
    assert lo < 5000.0 < hi
    assert hi - lo == pytest.approx(2 * 0.05 * 5000.0)


def test_the_teff_axis_covers_exactly_what_is_on_the_chart(built):
    """
    Given the Teff axis is set from the drawn content rather than autoscaled,
    When the spec is built,
    Then it spans the marks (error bars included) and the in-window track,
    padded, and is stated ascending because plotrender applies `x_range`
    BEFORE `x_inverted`.
    """
    # Arrange
    system, _ = built
    spec = _kiel_spec(system, system.active_components["evolutionarymodel"])
    track, mist_point, fit_point = spec.traces

    on_chart = np.concatenate(
        [
            np.atleast_1d(track.x),
            np.atleast_1d(mist_point.x),
            np.atleast_1d(fit_point.x) - np.atleast_1d(fit_point.xerr),
            np.atleast_1d(fit_point.x) + np.atleast_1d(fit_point.xerr),
        ]
    )
    pad = KIEL_X_PAD_FRAC * (on_chart.max() - on_chart.min())

    # Act
    lo, hi = spec.meta["x_range"]

    # Assert
    assert lo < hi
    assert lo == pytest.approx(on_chart.min() - pad)
    assert hi == pytest.approx(on_chart.max() + pad)


def test_the_teff_axis_ignores_track_rows_the_logg_window_clips_away(
    model_root,
):
    """
    Given a track that continues past the logg window -- a main-sequence
    star's arc runs on down the red giant branch to ~3000 K at logg < 3,
    When the Teff axis is set,
    Then those rows do not stretch it.  They are drawn but clipped away, and
    letting them autoscale the axis is what left more than half of a
    HAT-P-3 panel as whitespace.
    """
    # Arrange -- swap in a track whose last in-EEP-window row is a cool giant.
    # Only _track_curve reads _grids; the marks come from the compiled node.
    system, comp = _kiel_system(model_root)
    df = _track_table(
        [(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 0.5)],
        eeps=(300, 454, 605),
    )
    cool_giant = df["EEP"] == 605
    df.loc[cool_giant, "radius_mist"] = 100.0  # logg ~ 0.4
    df.loc[cool_giant, "teff_mist"] = 3000.0
    comp._grids = [mist_grid._assemble_grid(df)]

    # Act
    spec = _kiel_spec(system, comp)
    lo, hi = spec.meta["x_range"]

    # Assert -- drawn, but not on the axis
    assert 3000.0 in set(np.atleast_1d(spec.traces[0].x).tolist())
    assert lo > 3000.0
    assert spec.meta["y_range"] == list(KIEL_LOGG_WINDOW)
